import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(root_path)


import numpy as np
import tensorflow as tf
import argparse
import gpflow
from gpflow.params import Parameter
from gpflow import settings
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from core.svgp.multi_gpu import MultiGPU
from core.svgp.svgp import SVGP
from core.svgp.features import DFT_REAL_MATRIX, SingleInducingPoints
from core.utils.utils import InducingPoints
from utils.logging import get_logger
from utils.model_utils import data_iterator
from utils.utils import initialize_gpflow_graph, restore_model, save_model
from utils.common_utils import train_epoch, test_epoch


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='svgp', nargs='?', type=str)
parser.add_argument("--kern", default='matern32', type=str)
parser.add_argument("--dataset", default='elevation', type=str)
parser.add_argument("--J", type=int, default=12)
parser.add_argument("--split", default=0, type=int)

parser.add_argument("--iterations", type=int, default=100000)
parser.add_argument("-M", "--num_inducing", type=int, default=10)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-bs", "--batch-size", type=int, default=256)
parser.add_argument("--test-batch-size", type=int, default=200)

parser.add_argument("--num-gpu", type=int, default=1)

parser.add_argument("-run", "--run", type=str, default='debug')
parser.add_argument("--ckpt-dir", type=str)
args = parser.parse_args()

str_ = 'elevation/{}_{}_{}/J{}/bs{}_lr{}/NG{}_{}_split{}'.format(
    args.dataset, args.model, args.kern, args.num_inducing,
    args.J, args.batch_size, args.learning_rate,
    args.num_gpu, args.run, args.split
)
logger = get_logger('elevation', 'results/'+str_, __file__)
print = logger.info
print(args.__dict__)



class LongitudeFeatures(SingleInducingPoints):
    # the HVGP feature by moving a point eastwards.
    def __init__(self, z, num_translate, step, j, name='longitude feature'):
        gpflow.features.InducingFeature.__init__(self, name=name)
        self.Z = Parameter(z, dtype=settings.float_type)

        self.num_translate = num_translate
        self.step = step
        self.groups_per_direction = int(np.floor(num_translate / 2.) + 1)
        self.T = self.groups_per_direction

        self.j = j
        self._setup_DFT_matrix()
        self.Z.build()
        self.Z_orbit = self.orbit(self.Z.constrained_tensor)

    @gpflow.params_as_tensors
    def orbit(self, Z):
        # Z: [M, 2]
        step = tf.constant([self.step, 0.], dtype=settings.float_type)
        nums = tf.constant(np.arange(self.num_translate), dtype=settings.float_type)
        return Z + step[None, :] * nums[:, None, None] # [T, M, 2]

    def _setup_DFT_matrix(self):
        REAL = DFT_REAL_MATRIX(self.num_translate)
        RES = 2*REAL
        RES[0] = REAL[0]
        if self.num_translate % 2 == 0:
            RES[self.groups_per_direction-1] = REAL[self.groups_per_direction-1]
        self.DFT_matrix = RES[:self.groups_per_direction] # [groups_per_direction, num_translate]


class EarthKernel(gpflow.kernels.Kernel):
    """The kernel defined between the (long, lati) pairs."""
    def __init__(self, base_kern, name='wrap'):
        super().__init__(input_dim=2, name=name)
        self.base_kern = base_kern

    @staticmethod
    def coordinate(long_lati):
        # [-180, 180], [-90, 90]
        long, lati = long_lati[..., 0], long_lati[..., 1]
        long = long / 180. * np.pi # [-pi, pi]
        lati = lati / 180. * np.pi # [-pi/2, pi/2]

        x = tf.sin(np.pi / 2. - lati) * tf.cos(long)
        y = tf.sin(np.pi / 2. - lati) * tf.sin(long)
        z = tf.cos(np.pi / 2. - lati)

        return tf.concat([x[..., None], y[..., None], z[..., None]], -1)

    def K(self, X1, X2=None):
        fX1 = self.coordinate(X1)
        if X2 is None: fX2 = fX1
        else: fX2 = self.coordinate(X2)
        return self.base_kern.K(fX1, fX2)

    def Kdiag(self, X):
        fX = self.coordinate(X)
        return self.base_kern.Kdiag(fX)


def gen_dataset(args):
    with open(osp.join(root_path, 'data', 'ETOPO.npz'), 'rb') as file:
        data = np.load(file)
        longitude, latitude, elevation = data['longitude'], data['latitude'], data['height']
        X = np.concatenate([longitude[..., None], latitude[..., None]], axis=1)
        y = elevation[..., None]
    mean_y, std_y = elevation.mean(), elevation.std()
    y = (y - mean_y) / std_y
    N = longitude.shape[0]
    n_train = int(N * 0.8)
    perm = np.random.RandomState(0).permutation(longitude.shape[0])
    train_x, train_y = X[perm[:n_train]], y[perm[:n_train]]
    test_x, test_y = X[perm[n_train:]], y[perm[n_train:]]
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=np.random.RandomState(args.split))

    print('train_x {} | test_x {} | valid_x {}'.format(train_x.shape, test_x.shape, valid_x.shape))
    return train_x, valid_x, test_x, train_y, valid_y, test_y, mean_y, std_y, X, y


def get_model(args, train_x, train_y):
    kern = EarthKernel(gpflow.kernels.RBF(3, ARD=False, name='rbf'))
    lik = gpflow.likelihoods.Gaussian(variance=0.1, name='like')
    num_data = train_x.shape[0]
    tf_train_x, tf_train_y = data_iterator(train_x, train_y, min(args.batch_size, num_data))

    Z = train_x[np.random.permutation(train_x.shape[0])[:args.num_inducing]]
    if args.model == 'svgp':
        feat = InducingPoints(Z, name='features')
        model = SVGP(tf_train_x, tf_train_y, kern, lik, feat=feat, XY_tensor=True,
                     num_latent=1, num_data=num_data, name='model')
    elif args.model == 'translate_mg':
        step = 360. / args.J
        T = 1 + int(np.floor(args.J / 2.))
        feat_fn = lambda z, i, name: LongitudeFeatures(z, num_translate=args.J, step=step, j=i, name=name)
        model = MultiGPU(tf_train_x, tf_train_y, kern, lik, feat_fn, [Z]*T,
                         num_gpu=4, num_latent=1, num_data=num_data, XY_tensor=True, name='model')
    else:
        raise NotImplementedError

    if isinstance(model, MultiGPU):
        loss, grads_and_vars = model.build_objective_and_grads()
        obs_var = lik.variance.constrained_tensor
        print_dict = {'loss': loss, 'obs_var': obs_var}

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars)
        return model, print_dict, train_op, global_step, loss
    else:
        loss = model.objective / num_data
        lld, kl = model.lld, model.kl
        var_ = lik.variance.constrained_tensor
        print_dict = {'loss': loss, 'lld': lld, 'kl': kl, 'var': var_}
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return model, print_dict, train_op, global_step, loss

def make_pred(model, input_dim):
    x_pred = tf.placeholder(gpflow.settings.float_type, shape=[None, input_dim])
    y_pred = tf.placeholder(gpflow.settings.float_type, shape=[None, 1])
    pred_f_mean, pred_f_var = model._build_predict(x_pred)
    pred_m, pred_v = model.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
    pred_rmse_sq = (pred_m - y_pred) ** 2.
    pred_lld = model.likelihood.predict_density(pred_f_mean, pred_f_var, y_pred)
    return x_pred, y_pred, pred_rmse_sq, pred_lld, pred_m, pred_v

def load_result_logs(begin_epoch):
    if begin_epoch > 1:
        assert osp.exists(osp.join('results/', str_, 'res.npz'))
        with open(osp.join('results/', str_, 'res.npz'), 'rb') as f:
            results_logs = dict(np.load(f))
            results_logs = {k: v.tolist() for k,v in results_logs.items()}
    else:
        results_logs = {'train_loss': [], 'time_per_iter': [], # 'byte_in_use': [],
                        'valid_rmse_sq': [], 'valid_lld': [],
                        'test_rmse_sq': [], 'test_lld': []}
    return results_logs


def main():
    train_x, valid_x, test_x, train_y, valid_y, test_y, mean_y, std_y, X, y = gen_dataset(args)
    model, print_dict, train_op, global_step, loss = get_model(args, train_x, train_y)
    x_pred, y_pred, pred_rmse_sq, pred_lld, pred_m, pred_v = make_pred(model, input_dim=train_x.shape[1])
    valid_dict = {'valid_rmse_sq': pred_rmse_sq, 'valid_lld': pred_lld}
    test_dict = {'test_rmse_sq': pred_rmse_sq, 'test_lld': pred_lld}

    plot_lon, plot_lat = np.linspace(-180., 180., 720), np.linspace(-90., 90., 360)
    plot_X, plot_Y = np.meshgrid(plot_lon, plot_lat)
    plot_XY = np.concatenate([plot_X.reshape([-1, 1]), plot_Y.reshape([-1, 1])], 1)

    summary_op, summary_writer = tf.no_op(), None
    sess = initialize_gpflow_graph(model)
    saver = tf.train.Saver(max_to_keep=1)
    begin_epoch = restore_model(args, print, saver, sess, args.ckpt_dir)
    results_logs = load_result_logs(begin_epoch)

    iter_per_epoch = 100
    epochs, time_per_iter = int(args.iterations / iter_per_epoch), None
    args.epochs = epochs
    for epoch in range(begin_epoch, epochs+1):
        if epoch % 100 == 0 and epoch > begin_epoch:
            save_model(args, print, saver, sess, args.ckpt_dir, epoch)
            with open(osp.join('results/', str_, 'res.npz'), 'wb') as f:
                np.savez(f, **results_logs)

        res_avg, time_per_iter = train_epoch(args, print, print_dict, train_op, summary_op, summary_writer,
                                     global_step, sess, epoch, iter_per_epoch * args.batch_size)
        results_logs['train_loss'].append(res_avg['loss'])
        results_logs['time_per_iter'].append(time_per_iter)

        if epoch % 100 == 0 or epoch == epochs:
            train_epoch(args, print, print_dict, train_op, summary_op, summary_writer,
                                     global_step, sess, epoch, iter_per_epoch * args.batch_size)

            valid_res = test_epoch(args, print, x_pred, y_pred, valid_dict, summary_writer,
                             sess, epoch, global_step, valid_x, valid_y, PREFIX='VALID DATA', print_res=True)

            test_res = test_epoch(args, print, x_pred, y_pred, test_dict, summary_writer,
                             sess, epoch, global_step, test_x, test_y, PREFIX='TEST DATA', print_res=False)

            for key, value in {**valid_res, **test_res}.items():
                results_logs[key].append(value)

            plot_res = test_epoch(args, print, x_pred, None, {'m': pred_m, 'v': pred_v}, summary_writer,
                                  sess, epoch, global_step, plot_XY, None, PREFIX='PLOT DATA', print_res=False,
                                  return_all_averages=True)
            with open(osp.join('results/', str_, 'pred.npz'), 'wb') as f:
                np.savez(f, pred_m=plot_res[1]['m'], pred_v=plot_res[1]['v'], plot_XY=plot_XY)

    save_model(args, print, saver, sess, args.ckpt_dir, epochs+1)
    with open(osp.join('results/', str_, 'res.npz'), 'wb') as f:
        np.savez(f, **results_logs)


def plot_elevation():
    import cmocean
    import numpy as np
    import xarray as xr

    def init_plotting():
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["figure.figsize"] = [12, 6]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 1.4 * plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 1.7 * plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 1.2 * plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 1.3 * plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 1.3 * plt.rcParams['font.size']

    init_plotting()

    with open('/h/ssy/codes/data/ETOPO/data.npz', 'rb') as file:
        data = np.load(file)
        elevation = data['height']
        y = elevation[..., None]
    mean_y, std_y = elevation.mean(), elevation.std()

    plot_lon, plot_lat = np.linspace(-180., 180., 720), np.linspace(-90., 90., 360)
    svgp_M_path = 'results/elevation/elevation_svgp_matern32/J1000/bs12_lr256/NG0.01_1_splitnormal/pred.npz'
    ref_24M_path = 'results/elevation/elevation_translate_mg_matern32/J1000/bs24_lr256/NG0.01_1_splitnormal/pred.npz'

    fig, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [5, 6]}, sharey=True)

    with open(svgp_M_path, 'rb') as file:
        data = np.load(file)
        m = data['pred_m']
        m = mean_y + m * std_y

    da = xr.DataArray(
        data=m.reshape([len(plot_lat), len(plot_lon)]),
        dims=["latitude", "longitude", ],
        coords=dict(
            latitude=plot_lat,
            longitude=plot_lon,
        ),
    )
    da.plot.pcolormesh(cmap=cmocean.cm.balance, ax=axes[0], add_colorbar=False)

    with open(ref_24M_path, 'rb') as file:
        data = np.load(file)
        m = data['pred_m']
        m = mean_y + m * std_y

    da = xr.DataArray(
        data=m.reshape([len(plot_lat), len(plot_lon)]),
        dims=["latitude", "longitude", ],
        coords=dict(
            latitude=plot_lat,
            longitude=plot_lon,
        ),
    )
    da.plot.pcolormesh(cmap=cmocean.cm.balance, cbar_kwargs=dict(pad=0.01, aspect=30), ax=axes[1], add_labels=False,)
    axes[1].set_xlabel('longitude')

    plt.tight_layout()
    plt.savefig('figures/elevation.pdf', dvi=1200)

def bash():
    experiments = []
    M = 1000
    experiments.append('python exp/elevation/elevation.py -M %d --model translate_mg --J 12 ' % 1000)
    experiments.append('python exp/elevation/elevation.py -M %d --model translate_mg --J 24 ' % 1000)
    experiments.append('python exp/elevation/elevation.py -M %d  --model svgp' % 1000)
    experiments.append('python exp/elevation/elevation.py -M %d  --model svgp' % 3000)
    experiments.append('python exp/elevation/elevation.py -M %d  --model svgp' % 5000)


if __name__ == '__main__':
    bash()
    # main()
    # plot_elevation()

import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(root_path)

import numpy as np
import tensorflow as tf
import argparse
import gpflow
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from core.svgp.svgp import SVGP
from core.svgp.blk_svgp import BlockSVGP
from core.utils.utils import InducingPoints
from exp.transform_mnist.image_features import ImageFlipFeatures, ImageTranslateFeatures, ShareImageTranslateFeatures
from core.svgp.reflection import ReflectionFeatures
from utils.logging import get_logger
from utils.model_utils import data_iterator
from utils.utils import initialize_gpflow_graph, restore_model, save_model
from utils.common_utils import train_epoch, test_epoch


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='svgp', nargs='?', type=str)
parser.add_argument("--kern", default='matern32', type=str)
parser.add_argument("--dataset", default='translatemnist', type=str)
parser.add_argument("--J", type=int, default=0)
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

str_ = '{}/{}_{}_{}/J{}/bs{}_lr{}/NG{}_{}_split{}'.format(
    args.dataset, args.model, args.kern, args.num_inducing,
    args.J, args.batch_size, args.learning_rate,
    args.num_gpu, args.run, args.split
)
logger = get_logger('flipmnist', 'results/'+str_, __file__)
print = logger.info
print(args.__dict__)


def gen_dataset(args):
    with open(osp.join(root_path, 'data', '%s.npz' % args.dataset), 'rb') as file:
        data = np.load(file)
        train_x, train_y = data['train_x'], data['train_y']
        test_x, test_y = data['test_x'], data['test_y']
    train_x, test_x = train_x - 0.5, test_x - 0.5

    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=np.random.RandomState(args.split))
    return train_x, valid_x, test_x, train_y, valid_y, test_y


def get_model(args, train_x, train_y):
    kern = gpflow.kernels.RBF(train_x.shape[1], ARD=False, name='rbf')
    lik = gpflow.likelihoods.MultiClass(num_classes=10, name='like')
    num_data = train_x.shape[0]
    tf_train_x, tf_train_y = data_iterator(train_x, train_y, min(args.batch_size, num_data))

    Z = train_x[np.random.permutation(train_x.shape[0])[:args.num_inducing]]
    if args.model == 'svgp':
        feat = InducingPoints(Z, name='features')
        model = SVGP(tf_train_x, tf_train_y, kern, lik, feat=feat, XY_tensor=True,
                     num_latent=10, num_data=num_data, name='model')
    elif args.model == 'flip':
        feat = ImageFlipFeatures([Z] * args.J, patch_shape=[28, 28, 1], name='feature')
        model = BlockSVGP(tf_train_x, tf_train_y, kern, lik, feat, XY_tensor=True,
                          num_latent=10, num_data=num_data, name='model')
    elif args.model == 'negation':
        feat = ReflectionFeatures([Z] * args.J, name='feature')
        model = BlockSVGP(tf_train_x, tf_train_y, kern, lik, feat, XY_tensor=True,
                          num_latent=10, num_data=num_data, name='model')
    elif args.model == 'translate':
        feat = ImageTranslateFeatures([Z] * (4*4), num_translate=7, step=4, patch_shape=[28, 28, 1], name='feature')
        model = BlockSVGP(tf_train_x, tf_train_y, kern, lik, feat, XY_tensor=True,
                          num_latent=10, num_data=num_data, name='model')
    elif args.model == 'translate_share':
        feat = ShareImageTranslateFeatures(Z, num_translate=7, step=4, patch_shape=[28, 28, 1], name='feature')
        model = BlockSVGP(tf_train_x, tf_train_y, kern, lik, feat, XY_tensor=True,
                          num_latent=10, num_data=num_data, name='model')
    else:
        raise NotImplementedError


    loss = model.objective / num_data
    lld, kl = model.lld, model.kl
    print_dict = {'loss': loss, 'lld': lld, 'kl': kl}
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return model, print_dict, train_op, global_step

def make_pred(model, input_dim):
    x_pred = tf.placeholder(gpflow.settings.float_type, shape=[None, input_dim])
    y_pred = tf.placeholder(tf.int64, shape=[None, 1])
    pred_m, pred_v = model._build_predict(x_pred)
    pred_lld = model.likelihood.predict_density(pred_m, pred_v, y_pred)
    pred_acc = tf.to_float(tf.equal(tf.math.argmax(pred_m, -1), tf.squeeze(y_pred, 1)))
    return x_pred, y_pred, pred_acc, pred_lld

def load_result_logs(begin_epoch):
    if begin_epoch > 1:
        assert osp.exists(osp.join('results/', str_, 'res.npz'))
        with open(osp.join('results/', str_, 'res.npz'), 'rb') as f:
            results_logs = dict(np.load(f))
            results_logs = {k: v.tolist() for k,v in results_logs.items()}
    else:
        results_logs = {'train_loss': [], 'time_per_iter': [], # 'byte_in_use': [],
                        'valid_pred_acc': [], 'valid_lld': [],
                        'test_pred_acc': [], 'test_lld': []}
    return results_logs


def main():
    train_x, valid_x, test_x, train_y, valid_y, test_y = gen_dataset(args)
    model, print_dict, train_op, global_step = get_model(args, train_x, train_y)
    x_pred, y_pred, pred_acc, pred_lld = make_pred(model, input_dim=train_x.shape[1])
    valid_dict = {'valid_pred_acc': pred_acc, 'valid_lld': pred_lld}
    test_dict = {'test_pred_acc': pred_acc, 'test_lld': pred_lld}

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

        if epoch % 10 == 0 or epoch == epochs:
            valid_res = test_epoch(args, print, x_pred, y_pred, valid_dict, summary_writer,
                             sess, epoch, global_step, valid_x, valid_y, PREFIX='VALID DATA', print_res=True)

            test_res = test_epoch(args, print, x_pred, y_pred, test_dict, summary_writer,
                             sess, epoch, global_step, test_x, test_y, PREFIX='TEST DATA', print_res=False)

            for key, value in {**valid_res, **test_res}.items():
                results_logs[key].append(value)

    save_model(args, print, saver, sess, args.ckpt_dir, epochs+1)
    with open(osp.join('results/', str_, 'res.npz'), 'wb') as f:
        np.savez(f, **results_logs)


def plot_translatemnist(M):
    def init_plotting():
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["figure.figsize"] = [12, 6]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 2.4 * plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 1.7 * plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 1.7 * plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 1.7 * plt.rcParams['font.size']

    init_plotting()
    LW = 6
    colors_black = ['#969696', '#252525']
    colors_blue = ['#a1d99b', '#31a354']
    colors_green = ['#9ecae1', '#3182bd']


    svgp_M_path = 'results/translatemnist/svgp_matern32_%d/J0/bs256_lr0.001/NG1_debug_split0/' % M
    svgp_4M_path = 'results/translatemnist/svgp_matern32_%d/J0/bs256_lr0.001/NG1_debug_split0/' % (16*M)
    flip_4M_path = 'results/translatemnist/translate_matern32_%d/J0/bs256_lr0.001/NG1_debug_split0/' % M
    flip_share_4M_path = 'results/translatemnist/translate_share_matern32_%d/J0/bs256_lr0.001/NG1_debug_split0/' % M
    neg_4M_path = 'results/translatemnist/negation_matern32_%d/J16/bs256_lr0.001/NG1_debug_split0/' % M

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    with open(osp.join(svgp_M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss))[:600], train_loss[:600], c=colors_black[0], lw=LW, label=r'$SVGP: %d$' % M)
        axes[1].plot(np.arange(len(test_acc))[:60] * 10, test_acc[:60], c=colors_black[0], lw=LW)

    with open(osp.join(svgp_4M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss))[:600], train_loss[:600], c=colors_black[1], lw=LW, label=r'$SVGP: %d$' % (16*M))
        axes[1].plot(np.arange(len(test_acc))[:60] * 10, test_acc[:60], c=colors_black[1], lw=LW)

    with open(osp.join(neg_4M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss))[:600], train_loss[:600], c=colors_blue[1], lw=LW, label=r'$NEG: 16 \times %d$' % M)
        axes[1].plot(np.arange(len(test_acc))[:60] * 10, test_acc[:60], c=colors_blue[1], lw=LW)

    with open(osp.join(flip_4M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss))[:600], train_loss[:600], c=colors_green[1], lw=LW, label=r'$TRAN: 16 \times %d$' % M)
        axes[1].plot(np.arange(len(test_acc))[:60] * 10, test_acc[:60], c=colors_green[1], lw=LW)

    with open(osp.join(flip_share_4M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss))[:600], train_loss[:600], c=colors_green[0], lw=LW, ls='--', label=r'$TRAN-S: 16 \times %d$' % M)
        axes[1].plot(np.arange(len(test_acc))[:60] * 10, test_acc[:60], c=colors_green[0], lw=LW)

    axes[0].set_title('train loss')
    axes[1].set_title('test accuracy')
    axes[0].set_xlabel('Epochs')
    axes[1].set_xlabel('Epochs')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig('figures/translatemnist_M%d.pdf' % M)



def plot_flipmnist(M):
    def init_plotting():
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["figure.figsize"] = [12, 6]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 2.4 * plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 1.7 * plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 1.7 * plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 1.7 * plt.rcParams['font.size']

    init_plotting()
    LW = 6
    colors_black = ['#969696', '#252525']
    colors_blue = ['#a1d99b', '#31a354']
    colors_green = ['#9ecae1', '#3182bd']

    svgp_M_path = 'results/flipmnist/svgp_matern32_100/J0/bs256_lr0.001/NG1_debug_split0/'
    svgp_4M_path = 'results/flipmnist/svgp_matern32_400/J0/bs256_lr0.001/NG1_debug_split0/'
    flip_4M_path = 'results/flipmnist/flip_matern32_100/J4/bs256_lr0.001/NG1_debug_split0/'
    neg_4M_path = 'results/flipmnist/negation_matern32_100/J4/bs256_lr0.001/NG1_debug_split0/'

    # fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    with open(osp.join(svgp_M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss)), train_loss, c=colors_black[0], lw=LW, label=r'$SVGP: %d$' % M)
        axes[1].plot(np.arange(len(test_acc)) * 10, test_acc, c=colors_black[0], lw=LW)

    with open(osp.join(svgp_4M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss)), train_loss, c=colors_black[1], lw=LW, label=r'$SVGP: %d$' % (4*M))
        axes[1].plot(np.arange(len(test_acc)) * 10, test_acc, c=colors_black[1], lw=LW)

    with open(osp.join(neg_4M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss)), train_loss, c=colors_blue[1], lw=LW, label=r'$NEG: 4 \times %d$' % M)
        axes[1].plot(np.arange(len(test_acc)) * 10, test_acc, c=colors_blue[1], lw=LW)

    with open(osp.join(flip_4M_path, 'res.npz'), 'rb') as file:
        res = np.load(file)
        train_loss, test_acc, test_lld = res['train_loss'], res['test_pred_acc'], res['test_lld']
        axes[0].plot(range(len(train_loss)), train_loss, c=colors_green[1], lw=LW, label=r'$FLIP: 4 \times %d$' % M)
        axes[1].plot(np.arange(len(test_acc)) * 10, test_acc, c=colors_green[1], lw=LW)

    axes[0].set_ylim([1, 2])
    axes[1].set_ylim([0.8, 0.95])

    axes[0].set_title('train loss')
    axes[1].set_title('test accuracy')
    axes[0].set_xlabel('Epochs')
    axes[1].set_xlabel('Epochs')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig('figures/flipmnist_M%d.pdf' % M)


def bash_flipmnist():
    experiments = []
    M = 100
    experiments.append('python exp/transform_mnist/mnist.py --dataset flipmnist -M %d --J 4 --model flip' % M)
    experiments.append('python exp/transform_mnist/mnist.py --dataset flipmnist -M %d --J 4 --model negation' % M)
    experiments.append('python exp/transform_mnist/mnist.py --dataset flipmnist -M %d  --model svgp' % M)
    experiments.append('python exp/transform_mnist/mnist.py --dataset flipmnist -M %d  --model svgp' % (4*M))


def bash_translatemnist():
    experiments = []
    M = 50
    experiments.append('python exp/transform_mnist/mnist.py -M %d --J 16 --model translate' % M)
    experiments.append('python exp/transform_mnist/mnist.py -M %d --J 16 --model translate_share' % M)
    experiments.append('python exp/transform_mnist/mnist.py -M %d --J 16 --model negation' % M)
    experiments.append('python exp/transform_mnist/mnist.py -M %d  --model svgp' % M)
    experiments.append('python exp/transform_mnist/mnist.py -M %d  --model svgp' % (16*M))


if __name__ == '__main__':
    main()
    # bash_translatemnist()
    # bash_flipmnist()
    # plot_flipmnist(100)
    # plot_translatemnist(50)
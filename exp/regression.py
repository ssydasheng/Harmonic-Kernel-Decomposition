import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import tensorflow as tf
import argparse
import gpflow
from sklearn.model_selection import train_test_split

from utils.data import get_regression_data
from utils.logging import get_logger
from utils.model_utils import get_model
from utils.utils import initialize_gpflow_graph, restore_model, save_model
from utils.common_utils import train_epoch, test_epoch


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='svgp', nargs='?', type=str)
parser.add_argument("--kern", default='matern32', type=str)
parser.add_argument("--J", type=int, default=0)
parser.add_argument("--dataset", default='kin8nm', nargs='?', type=str)
parser.add_argument("--split", default=0, type=int)

parser.add_argument("--iterations", type=int, default=30000)
parser.add_argument("-M", "--num_inducing", type=int, default=10)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.003)
parser.add_argument("-bs", "--batch-size", type=int, default=256)
parser.add_argument("--test-batch-size", type=int, default=200)

parser.add_argument("--wrap-nh", type=int, default=1)
parser.add_argument("--wrap-nu", type=int, default=500)
parser.add_argument("--wrap-dim", type=int, default=8)

parser.add_argument("--num-gpu", type=int, default=1)

parser.add_argument("-run", "--run", type=str, default='debug')
parser.add_argument("--ckpt-dir", type=str)
args = parser.parse_args()

str_ = 'regression/{}/{}_{}_{}/J{}/bs{}_lr{}/NG{}_{}_split{}'.format(
    args.dataset, args.model, args.kern, args.num_inducing,
    args.J, args.batch_size, args.learning_rate,
    args.num_gpu, args.run, args.split
)
logger = get_logger('regression', 'results/'+str_, __file__)
print = logger.info
print(args.__dict__)


def gen_dataset(args):
    data = get_regression_data(args.dataset, split=args.split, prop=0.8)
    train_x, train_y, test_x, test_y = data.X_train, data.Y_train, data.X_test, data.Y_test
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=np.random.RandomState(args.split))
    return train_x, valid_x, test_x, train_y, valid_y, test_y


def make_pred(model, input_dim):
    x_pred = tf.placeholder(gpflow.settings.float_type, shape=[None, input_dim])
    y_pred = tf.placeholder(gpflow.settings.float_type, shape=[None, 1])
    pred_f_mean, pred_f_var = model._build_predict(x_pred)
    pred_m, pred_v = model.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)
    pred_rmse_sq = (pred_m - y_pred) ** 2.
    pred_lld = model.likelihood.predict_density(pred_f_mean, pred_f_var, y_pred)
    return x_pred, y_pred, pred_rmse_sq, pred_lld

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
    train_x, valid_x, test_x, train_y, valid_y, test_y = gen_dataset(args)
    model, print_dict, train_op, global_step = get_model(args, train_x, train_y)
    x_pred, y_pred, pred_rmse_sq, pred_lld = make_pred(model, input_dim=train_x.shape[1])
    valid_dict = {'valid_rmse_sq': pred_rmse_sq, 'valid_lld': pred_lld}
    test_dict = {'test_rmse_sq': pred_rmse_sq, 'test_lld': pred_lld}

    summary_op, summary_writer = tf.no_op(), None
    sess = initialize_gpflow_graph(model)
    saver = tf.train.Saver(max_to_keep=1)
    begin_epoch = restore_model(args, print, saver, sess, args.ckpt_dir)
    results_logs = load_result_logs(begin_epoch)

    iter_per_epoch = 100
    epochs, time_per_iter = int(args.iterations / iter_per_epoch), None
    args.epochs = epochs
    # byte_in_use = tf.contrib.memory_stats.BytesInUse()
    for epoch in range(begin_epoch, epochs+1):
        if epoch % 10 == 0 and epoch > begin_epoch:
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

def bash():
    cmds = []
    M = 1000
    J = 8
    prefix = 'python exp/regression.py'
    for dataset in ['trueyear', 'wilson_3droad', 'wilson_song', 'wilson_buzz', 'wilson_houseelectric']:
        for split in [0, 1, 2]:
            # single GPU
            cmds.append(prefix + ' --model svgp -M %d --dataset %s  --kern matern32 --split %d' % (M, dataset, split))
            cmds.append(prefix + ' --model svgp -M %d --dataset %s  --kern matern32 --split %d' % (2*M, dataset, split))

            # multiple GPUs
            cmds.append(prefix + ' --model refmg -M %d --J %d --num-gpu %d --dataset %s --kern matern32 --split %d'%(M, J, J, dataset, split))
            cmds.append(prefix + ' --model refmg -M %d --J %d --num-gpu %d --dataset %s --kern matern32 --split %d'%(2*M, J, J, dataset, split))

if __name__ == '__main__':
    main()

import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split

from gpflow.likelihoods import MultiClass
from gpflow import settings

from core.dgp.multigpu.dgp import MultiGPU_DGP
from core.dgp.utils.init_conv_layer import register_conv_layers
from core.dgp.utils.utils import SoftmaxMultiClass
from utils.utils import initialize_gpflow_graph
from utils.logging import get_logger
from utils.common_utils import train_epoch, test_epoch
from utils.dataset import load_cifar10
from utils.net_utils import networks
from utils.net_utils import DENSE_LAYER_DICT, CONV_LAYER_DICT, MODEL_DICT
from utils.utils import save_model, restore_model


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('-m', '--method', type=str)
parser.add_argument('-n', '--network', type=str, default='L2-1')

parser.add_argument('--ard', action='store_true')
parser.add_argument('--white', action='store_true', default=False)
parser.add_argument('--float32', action='store_true', default=False)
parser.add_argument('-bs', '--batch_size', type=int, default=64)

parser.add_argument('-tbs', '--test_batch_size', type=int, default=64)
parser.add_argument('--test_every', type=int, default=10)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.003)
parser.add_argument('-e', '--epochs', type=int, default=384)

parser.add_argument('-run', '--run', type=str, default='exp')
parser.add_argument('-n_runs', '--n_runs', type=int, default=1)
parser.add_argument('-seed', '--init_seed', type=int, default=1)
parser.add_argument("--ckpt-dir", type=str)
args = parser.parse_args()

task_name = 'classification/%s' % args.dataset
str_ = task_name + '/f32{}/{}_{}_W{}_ard{}/BS{}_LR{}/{}_IS{}/'.format(
    args.float32, args.method, args.network, args.white, args.ard,
    args.batch_size, args.learning_rate,
    args.run, args.init_seed)
logger = get_logger('CONV', 'results/'+str_, __file__)
print = logger.info
print(args.__dict__)
np.random.seed(args.init_seed)
tf.set_random_seed(args.init_seed)
if args.float32:
    settings.set_float_type('float32')
    settings.set_jitter(1e-4)
else:
    settings.set_float_type('float64')


def get_multiclass(args):
    return MultiClass(10, name='like')


def make_dgp(train_x, train_y, N, input_dim, XY_tensor, numpy_train_x):
    with tf.variable_scope('dgp'):
        net = networks(args.network)
        layers = register_conv_layers(numpy_train_x.astype(settings.float_type), net['kerns'], net['inducing_features'],
                                      feature_maps=net['feature_maps'],
                                      filter_sizes=net['filter_sizes'], strides=net['strides'],
                                      pools=net['pools'], pool_sizes=net['pool_sizes'],
                                      num_outputs=10, white=args.white, Layer=DENSE_LAYER_DICT[args.method],
                                      q_sqrt_init_scale=1e-5,
                                      num_inducings=net['nms'], paddings=net['paddings'],
                                      ARD=args.ard, convLayer=CONV_LAYER_DICT[args.method])
        if XY_tensor:
            model = MODEL_DICT[args.method](
                tf.reshape(train_x, [tf.shape(train_x)[0], input_dim]), train_y, layers, get_multiclass(args),
                num_latent=10, minibatch_size=args.batch_size, num_data=N, XY_tensor=True,
                integrate_likelihood=True, num_samples=1, name='DGP')
        else:
            model = MODEL_DICT[args.method](
                train_x.reshape([N, input_dim]), train_y, layers, get_multiclass(args),
                num_latent=10, minibatch_size=args.batch_size, num_data=N, XY_tensor=False,
                integrate_likelihood=True, num_samples=1, name='DGP')

    if isinstance(model, MultiGPU_DGP):
        elbo, grad_and_vars = model.build_objective_and_grads()
        normalized_logll, normalized_kl = model.elbo_logll / N, model.elbo_kl / N
        loss = -elbo / N
        print_dict = {'loss': loss, 'elbo_logll': normalized_logll, 'elbo_kl': normalized_kl}

        global_step = tf.Variable(0., trainable=False)
        lr = tf.train.exponential_decay(args.learning_rate, global_step, 50000, 0.25, staircase=True)
        train_op = tf.train.AdamOptimizer(lr).apply_gradients(grad_and_vars, global_step=global_step)
    else:
        loss = model.objective / N
        normalized_logll, normalized_kl = model.elbo_logll / N, model.elbo_kl / N
        print_dict = {'loss': loss, 'elbo_logll': normalized_logll, 'elbo_kl': normalized_kl}
        loss = model.objective
        global_step = tf.Variable(0., trainable=False)
        lr = tf.train.exponential_decay(args.learning_rate, global_step, 50000, 0.25, staircase=True)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
    return model, print_dict, global_step, train_op


def make_pred(model, input_dim):
    x_pred = tf.placeholder(settings.float_type, shape=[None, input_dim])
    y_pred = tf.placeholder(tf.int64, shape=[None, 1])
    pred_m, pred_v = model.predict_y(x_pred, num_samples=5)
    pred_m = tf.reduce_mean(pred_m, 0)
    pred_lld = model.predict_density(x_pred, y_pred, num_samples=5)
    pred_acc = tf.to_float(tf.equal(tf.math.argmax(pred_m, -1), tf.squeeze(y_pred, 1)))
    return x_pred, y_pred, pred_acc, pred_lld

def restore(args, print, saver, sess, results_logs):
    begin_epoch = restore_model(args, print, saver, sess, args.ckpt_dir)
    if begin_epoch > 1:
        with open(osp.join('results/', str_, 'res.npz'), 'rb') as f:
            res = np.load(f)
            for key in results_logs:
                results_logs[key].extend(res[key])
    return begin_epoch, results_logs

def main():
    numpy_train_x, numpy_train_y, test_x, test_y = load_cifar10(dtype='float64')
    mean_x = numpy_train_x.mean(0)
    numpy_train_x, test_x = numpy_train_x - mean_x, test_x - mean_x
    numpy_train_x, test_x = numpy_train_x.astype(settings.float_type), test_x.astype(settings.float_type)
    numpy_train_x, valid_x, numpy_train_y, valid_y = train_test_split(
        numpy_train_x, numpy_train_y, test_size=0.1, random_state=np.random.RandomState(args.init_seed))

    N, input_dim = numpy_train_x.shape[0], np.prod(numpy_train_x.shape[1:])

    model, print_dict, global_step, train_op = make_dgp(numpy_train_x, numpy_train_y, N, input_dim,
                                                        XY_tensor=False, numpy_train_x=numpy_train_x)

    x_pred, y_pred, pred_acc, pred_lld = make_pred(model, input_dim)
    valid__dict = {'valid_acc': pred_acc, 'valid_lld': pred_lld}
    test_dict = {'test_acc': pred_acc, 'test_lld': pred_lld}

    results_logs = {'valid_acc': [], 'valid_lld': [],
                    'test_acc': [], 'test_lld': [], 'time_per_iter': []}
    summary_op, summary_writer = tf.no_op(), None
    sess = initialize_gpflow_graph(model)
    saver = tf.train.Saver(max_to_keep=1)
    begin_epoch, results_logs = restore(args, print, saver, sess, results_logs)

    for epoch in range(begin_epoch, args.epochs+1):
        # if epoch % 10 == 0:
        if epoch % 1 == 0 and epoch > begin_epoch:
            save_model(args, print, saver, sess, args.ckpt_dir, epoch)
            with open(osp.join('results/', str_, 'res.npz'), 'wb') as f:
                np.savez(f, **results_logs)
            if not isinstance(model, MultiGPU_DGP):
                for layer_idx, l in enumerate(model.layers):
                    l.feature.save_inducing(sess, osp.join('results/', str_, 'Z_%d.npz'% layer_idx))

        _, time_per_iter = train_epoch(args, print, print_dict, train_op, summary_op, summary_writer,
                                       global_step, sess, epoch, N)
        if epoch % args.test_every == 0 or epoch == args.epochs:
            valid_res, all_valid_averages = test_epoch(
                args, print, x_pred, y_pred, valid__dict, summary_writer, sess, epoch,
                global_step, valid_x.reshape([-1, input_dim]), valid_y, PREFIX='VALID DATA', return_all_averages=True, print_res=True)

            test_res, all_test_averages = test_epoch(
                args, print, x_pred, y_pred, test_dict, summary_writer, sess, epoch,
                global_step, test_x.reshape([-1, input_dim]), test_y, PREFIX='TEST DATA', return_all_averages=True, print_res=False)

            for key, value in {**valid_res, **test_res}.items():
                results_logs[key].append(value)
            results_logs['time_per_iter'].append(time_per_iter)

    with open(osp.join('results/', str_, 'res.npz'), 'wb') as f:
        np.savez(f, **results_logs)

def bash():
    seeds = [1, 2, 3]
    commands = [
        'python -m exp/classification -n L1-1 -m kfgd -run debug --white -tbs 10 -seed 1'
        'python -m exp/classification -n L1-2 -m kfgd -run debug --white -tbs 10 -seed 1'
        'python -m exp/classification -n L1-2 -m sovgd -run debug --white -tbs 10 -seed 1'
        'python -m exp/classification -n L1-1-2-v3 -m blkkfgd -run debug --white -tbs 10 -seed 1'
        'python -m exp/classification -n L1-1-mg4 -m mgkfgd -run debug --white -tbs 10 -seed 1'
        
        'python -m exp/classification -n L2-3 -m kfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L2-4 -m kfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L2-4 -m sovgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L2-3-2-v3 -m blkkfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L2-3-mg4 -m mgkfgd -run debug -tbs 10 -seed 1'
        
        'python -m exp/classification -n L3-4 -m kfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L3-21 -m kfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L3-21 -m sovgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L3-4-2-v3 -m blkkfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L3-4-mg4 -m mgkfgd -run debug -tbs 10 -seed 1'
        
        'python -m exp/classification -n L4-7 -m kfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L4-22 -m kfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L4-13 -m sovgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L4-7-2-v3 -m blkkfgd -run debug -tbs 10 -seed 1'
        'python -m exp/classification -n L4-7-mg4 -m mgkfgd -run debug -tbs 10 -seed 1'
    ]

if __name__ == '__main__':
    main()

import numpy as np
import gpflow
import tensorflow as tf

from core.svgp.svgp import SVGP
from core.svgp.blk_svgp import BlockSVGP
from core.svgp.multi_gpu import MultiGPU
from core.svgp.reflection import ReflectionFeatures, ReflectionSingleFeatures
from core.utils.utils import InducingPoints
from utils.utils import get_kemans_init, median_distance_global


def get_model(args, train_x, train_y):
    kern = get_kern(args, train_x)
    lik = gpflow.likelihoods.Gaussian(name='lik')
    lik.variance = 0.1
    num_data = train_x.shape[0]
    tf_train_x, tf_train_y = data_iterator(train_x, train_y, min(args.batch_size, num_data))
    Z = get_kemans_init(train_x, args.num_inducing)

    if args.model == 'svgp':
        feat = InducingPoints(Z, name='features')
        model = SVGP(tf_train_x, tf_train_y, kern, lik, feat=feat, XY_tensor=True, num_data=num_data, name='model')
    elif args.model == 'ref':
        feat = ReflectionFeatures([Z] * args.J, U=np.eye(Z.shape[-1]), name='feature')
        model = BlockSVGP(tf_train_x, tf_train_y, kern, lik, feat, XY_tensor=True, num_data=num_data, name='model')
    elif args.model == 'refmg':
        assert args.num_gpu > 1
        Sigma = np.cov(train_x, rowvar=False)
        _, U = np.linalg.eigh(Sigma)
        def feat_fn(z, i, name):
            return ReflectionSingleFeatures(z, U, i, args.J, name)
        model = MultiGPU(tf_train_x, tf_train_y, kern, lik, feat_fn, [Z]*args.J, args.num_gpu,
                         XY_tensor=True, num_data=num_data, name='model')
    else:
        raise NotImplementedError

    if isinstance(model, MultiGPU):
        loss, grads_and_vars = model.build_objective_and_grads()
        obs_var = lik.variance.constrained_tensor
        print_dict = {'loss': loss, 'obs_var': obs_var}

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars)
        return model, print_dict, train_op, global_step
    else:
        loss = model.objective / num_data
        obs_var = lik.variance.constrained_tensor
        print_dict = {'loss': loss, 'obs_var': obs_var}

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return model, print_dict, train_op, global_step


def get_kern(args, train_x):
    ls = median_distance_global(train_x)
    if args.kern == 'rbf':
        kern = gpflow.kernels.RBF(train_x.shape[1], lengthscales=ls, name='rbf')
    elif args.kern == 'matern32':
        kern = gpflow.kernels.Matern32(train_x.shape[1], lengthscales=ls, name='matern32')
    else:
        raise NotImplementedError
    return kern


def data_iterator(numpy_train_x, numpy_train_y, batch_size):
    iterator = tf.data.Dataset.from_tensor_slices(
        (numpy_train_x.astype(gpflow.settings.float_type),
         numpy_train_y.astype(gpflow.settings.float_type))).shuffle(buffer_size=10000)
    iterator = iterator.batch(batch_size, drop_remainder=True).repeat()
    train_x, train_y = iterator.make_one_shot_iterator().get_next()
    return train_x, train_y

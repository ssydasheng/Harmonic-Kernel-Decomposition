# Adapted from https://github.com/kekeblom/DeepCGP

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import settings, kernels, features
from gpflow.mean_functions import Zero
from gpflow.session_manager import get_default_session

from ..dense_layer import SVGP_KFGD_Layer
from ..conv_layer import Conv_KFGD_Layer
from .views import FullView
from .meanFunctions import Conv2dMean
from .img_utils import xavier_conv
from .kernels import ConvKernel, TICKernel


def register_conv_layer(base_kern, inducing_feature, NHWC_X, num_inducing, feature_map, filter_size, stride, pool, pool_size,
                        q_sqrt_init_scale=1., white=False, ARD=False,
                        padding="VALID", convLayer=Conv_KFGD_Layer, name='Layer'):
    NHWC = NHWC_X.shape
    view = FullView(input_size=NHWC[1:3],
                    filter_size=filter_size,
                    feature_maps=NHWC[3],
                    stride=stride,
                    padding=padding)

    conv_mean = Conv2dMean(filter_size, NHWC[3], feature_map, stride=stride, padding=padding, name=name)
    conv_mean.set_trainable(False)
    H_X = xavier_conv(NHWC_X, filter_size, NHWC[3], feature_map, stride, padding)

    conv_features = inducing_feature.from_images(NHWC_X, num_inducing, filter_size, name=name)
    patch_length = filter_size ** 2 * NHWC[3]
    kernel = kernels.RBF(patch_length, variance=5., lengthscales=5.0, ARD=ARD, name=name)

    if pool is not None:
        pool_func = dict(max=tf.layers.max_pooling2d, mean=tf.layers.average_pooling2d)[pool]
        tf_HX = pool_func(H_X, pool_size, pool_size, "SAME")
        H_X = get_default_session().run(tf_HX)

    conv_layer = convLayer(
        base_kern=kernel,
        mean_function=conv_mean,
        feature=conv_features,
        view=view,
        white=white,
        gp_count=feature_map,
        pool=pool,
        pool_size=pool_size,
        q_sqrt_init_scale=q_sqrt_init_scale,
        name=name)
    return conv_layer, H_X


def register_final_layer(base_kern, inducing_feature, NHWC_X, num_outputs, num_inducing, filter_size, stride,
                         white=False, ARD=False, padding="VALID", Layer=SVGP_KFGD_Layer, name='layer'):
    NHWC = NHWC_X.shape
    input_dim = filter_size ** 2 * NHWC[3]
    view = FullView(input_size=NHWC[1:],
                    filter_size=filter_size,
                    feature_maps=NHWC[3],
                    stride=stride,
                    padding=padding)
    mean_func = gpflow.mean_functions.Zero(output_dim=num_outputs)

    inducing = inducing_feature.from_images(NHWC_X, num_inducing, filter_size, name=name)
    patch_weights = None
    if base_kern == 'conv':
        kernel = ConvKernel(
            base_kern=gpflow.kernels.RBF(input_dim, variance=5., lengthscales=5., ARD=ARD, name=name),
            view=view, patch_weights=patch_weights)
    elif base_kern == 'tick':
        kernel = TICKernel(
            patch_kern=gpflow.kernels.RBF(input_dim, variance=5., lengthscales=5., ARD=ARD, name=name+'patch'),
            loc_kern=gpflow.kernels.Matern32(2, variance=1., lengthscales=3., ARD=ARD, name=name+'loc'),
            view=view, patch_weights=patch_weights, name=name)
    else:
        raise ValueError("Invalid last layer kernel")

    return Layer(kern=kernel,
                 num_outputs=num_outputs,
                 feature=inducing,
                 mean_function=mean_func,
                 white=white,
                 name=name)

def register_conv_layers(NHWC_X, base_kerns, inducing_features, num_outputs, num_inducings, feature_maps, strides, filter_sizes,
                         pools, pool_sizes, paddings=None, q_sqrt_init_scale=1., ARD=False, white=False,
                         Layer=SVGP_KFGD_Layer, convLayer=Conv_KFGD_Layer):
    assert len(strides) == len(filter_sizes)
    assert len(feature_maps) == (len(num_inducings) - 1)
    assert len(feature_maps) == len(pools)
    assert len(feature_maps) == len(pool_sizes)
    if paddings is None:
        paddings = ["VALID"] * len(base_kerns)

    layers = []
    X_running = NHWC_X
    for i, (base_kern, inducing_feature, M, feature_map, stride, filter_size, pool, pool_size, padding) in enumerate(zip(
            base_kerns[:-1], inducing_features[:-1], num_inducings[:-1],
            feature_maps, strides, filter_sizes, pools, pool_sizes, paddings[:-1])):
        conv_layer, X_running = register_conv_layer(
            base_kern, inducing_feature, X_running, M, feature_map, filter_size, stride,
            pool=pool, pool_size=pool_size, q_sqrt_init_scale=q_sqrt_init_scale,
            white=white, ARD=ARD,
            padding=padding, convLayer=convLayer, name='Layer_'+str(i))
        layers.append(conv_layer)
    layers.append(register_final_layer(
        base_kerns[-1], inducing_features[-1], X_running, num_outputs, num_inducings[-1], filter_sizes[-1], strides[-1],
        white=white, ARD=ARD, padding=paddings[-1], Layer=Layer, name='Layer_'+str(len(base_kerns))))
    return layers
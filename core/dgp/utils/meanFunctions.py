"""
Adapted from https://github.com/kekeblom/DeepCGP
"""

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.decors import params_as_tensors

def identity_conv2d(NHWC_X, filter_size, feature_maps_in, feature_maps_out=1, stride=1, padding="VALID"):
    filter = np.zeros((filter_size, filter_size,
                                feature_maps_in, feature_maps_out), dtype=gpflow.settings.float_type)
    filter[filter_size // 2, filter_size // 2, :, :] = 1.0 / feature_maps_in
    return tf.nn.conv2d(NHWC_X, filter, strides=[1, stride, stride, 1], padding=padding, data_format="NHWC")

def xavier_conv2d(NHWC_X, filter_size, feature_maps_in, feature_maps_out=1, stride=1, padding="VALID"):
    fan = feature_maps_in * filter_size**2.
    gain = 1.
    std = gain / np.sqrt(fan)
    bound = np.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    filter = np.random.uniform(-bound, bound, size=[filter_size, filter_size, feature_maps_in, feature_maps_out])
    return tf.nn.conv2d(NHWC_X, filter, strides=[1, stride, stride, 1], padding=padding, data_format="NHWC")
    # return tf.layers.conv2d(NHWC_X, feature_maps_out, filter_size,
    #                         strides=[stride, stride], padding="valid", data_format="channels_last", use_bias=False)

class IdentityConv2dMean(gpflow.mean_functions.MeanFunction):
    def __init__(self, filter_size, feature_maps_in, feature_maps_out=1, stride=1, padding="VALID", mark_used=False, name='conv2d'):
        super().__init__(name=name)
        self.filter_size = filter_size
        self.feature_maps_in = feature_maps_in
        self.feature_maps_out = feature_maps_out
        self.stride = stride
        self.padding = padding
        self.mark_used = mark_used
        self.conv_filter = gpflow.Param(self._init_filter())

    @params_as_tensors
    def _conv2d(self, NHWC_X):
        return tf.nn.conv2d(NHWC_X, self.conv_filter,
                strides=[1, self.stride, self.stride, 1],
                padding=self.padding,
                data_format="NHWC")

    def __call__(self, NHWC_X):
        ret = self._conv2d(NHWC_X)
        if self.mark_used:
            self.conv_filter.is_initialized_tensor.mark_used()
        return ret

    def _init_filter(self):
        identity_filter = np.zeros((self.filter_size, self.filter_size,
            self.feature_maps_in, self.feature_maps_out), dtype=gpflow.settings.float_type)
        identity_filter[self.filter_size // 2, self.filter_size // 2, :, :] = 1.0
        return identity_filter

class Conv2dMean(IdentityConv2dMean):
    """The first filter map copies the center most pixel in the input and the rest are zero mean."""
    def _init_filter(self):
        # Only supports square filters with odd size for now. The first feature maps copies the input,
        # the rest is zero mean.
        identity_filter = np.zeros((self.filter_size, self.filter_size,
            self.feature_maps_in, self.feature_maps_out), dtype=gpflow.settings.float_type)
        min_size = min(self.feature_maps_in, self.feature_maps_out)
        for idx in range(min_size):
            identity_filter[self.filter_size // 2, self.filter_size // 2, idx, idx] = 1.0
        return identity_filter

    def __call__(self, NHWC_X):
        value = super().__call__(NHWC_X)
        N = tf.shape(NHWC_X)[0]
        return tf.reshape(value, [N, -1])

class PatchwiseConv2d(Conv2dMean):
    def __init__(self, filter_size, feature_maps_in, out_height, out_width, name='patchconv'):
        super().__init__(filter_size, feature_maps_in, name=name)
        self.out_height = out_height
        self.out_width = out_width

    @params_as_tensors
    def __call__(self, PNL_patches):
        kernel = tf.reshape(self.conv_filter, [self.filter_size**2 * self.feature_maps_in,
            self.feature_maps_in])
        # def foreach_patch(NL_patches):
        #     return tf.matmul(NL_patches, kernel)
        # PN1_out = tf.map_fn(foreach_patch, PNL_patches)
        PN1_out = tf.matmul(PNL_patches, kernel)
        PNL = tf.shape(PNL_patches)
        return tf.reshape(tf.transpose(PN1_out, [2, 1, 0]), [PNL[1], PNL[0]])


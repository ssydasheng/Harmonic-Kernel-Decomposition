import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.kernels import Stationary

from .img_utils import kerne_by_conv
float_type = settings.float_type


def fast_kernel_by_conv(NHWC_X, Z, view, base_kern):
    N = tf.shape(NHWC_X)[0]
    output_size = view.out_image_height * view.out_image_width

    patch_square, z_square, patch_times_z = kerne_by_conv(NHWC_X, Z, view, base_kern)
    # [batch_size, output_size_0, output_size_1, num_inducing]
    sq_dist = patch_square + z_square - 2 * patch_times_z
    sq_dist = tf.reshape(sq_dist, [N, output_size, tf.shape(Z)[0]])
    NPM_Kuf = base_kern.K_r2(sq_dist)
    return NPM_Kuf


class MultiOutputConvKernel(gpflow.kernels.Kernel):
    def __init__(self, base_kern, view, patch_count, name='kernel'):
        input_dim = np.prod(view.input_size) * view.feature_maps
        super().__init__(input_dim=input_dim, name=name)
        self.base_kern = base_kern
        self.patch_count = patch_count
        self.view = view

    def Kdiag(self, NHWC_X):
        if isinstance(self.base_kern, Stationary):
            N = tf.shape(NHWC_X)[0]
            output_size = self.view.out_image_height * self.view.out_image_width
            return tf.fill([output_size, N], tf.squeeze(self.base_kern.variance))
        else:
            PNL_patches = self.view.extract_patches_PNL(NHWC_X)
            return self.base_kern.Kdiag(PNL_patches)


class ConvKernel(gpflow.kernels.Kernel):
    def __init__(self, base_kern, view, patch_weights=None, name='convK'):
        super().__init__(input_dim=np.prod(view.input_size), name=name)
        self.base_kern = base_kern
        self.view = view
        self.patch_length = view.patch_length
        self.patch_count = view.patch_count
        self.image_size = self.view.input_size
        if patch_weights is None or patch_weights.size != self.patch_count:
            patch_weights = np.ones(self.patch_count, dtype=settings.float_type)
        self.patch_weights = gpflow.Param(patch_weights, name=name)

    def _reshape_X(self, ND_X):
        ND = tf.shape(ND_X)
        return tf.reshape(ND_X, [ND[0]] + list(self.view.input_size))

    def Kdiag(self, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        patches = self.view.extract_patches(NHWC_X)
        w = self.patch_weights
        W = w[None, :] * w[:, None]
        return tf.reduce_sum(self.base_kern.K(patches) * W, [-1, -2]) / (self.patch_count ** 2)



class TICKernel(gpflow.kernels.Kernel):
    # Loosely based on https://github.com/markvdw/convgp/blob/master/convgp/convkernels.py
    def __init__(self, patch_kern, loc_kern, view, patch_weights=None, name='tick'):
        super().__init__(input_dim=np.prod(view.input_size), name=name)
        self.patch_kern = patch_kern
        self.loc_kern = loc_kern
        self.view = view
        self.patch_length = view.patch_length
        self.patch_count = view.patch_count
        self.image_size = self.view.input_size
        if patch_weights is None or patch_weights.size != self.patch_count:
            patch_weights = np.ones(self.patch_count, dtype=settings.float_type)
        self.patch_weights = gpflow.Param(patch_weights, name=name)

    def _reshape_X(self, ND_X):
        ND = tf.shape(ND_X)
        return tf.reshape(ND_X, [ND[0]] + list(self.view.input_size))

    def get_loc(self):
        H, C = self.view._out_image_size()
        X, Y = tf.meshgrid(np.asarray(range(H)).astype(settings.float_type),
                           np.asarray(range(C)).astype(settings.float_type))
        XY = tf.transpose(tf.concat([X[..., None], Y[..., None]], axis=-1), [1, 0, 2])
        return tf.reshape(XY, [H * C, 2]) * self.view.stride

    def Kdiag(self, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        patches = self.view.extract_patches(NHWC_X)
        locs = self.get_loc()
        w = self.patch_weights
        W = w[None, :] * w[:, None]

        patch_K = self.patch_kern.K(patches)
        loc_K = self.loc_kern.K(locs)
        K = patch_K * loc_K[None]
        return tf.reduce_sum(K * W, [-1, -2]) / (self.patch_count ** 2)
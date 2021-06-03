import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, params_as_tensors, name_scope
from gpflow.kernels import Stationary
from gpflow.params import Parameter, ParamList
from gpflow import params_as_tensors_for
from gpflow.dispatch import dispatch

from .img_utils import _cluster_patches_with_locations, kerne_by_conv
from core.utils.utils import InducingPoints
from .kernels import MultiOutputConvKernel, ConvKernel, TICKernel, fast_kernel_by_conv
from .feature import PatchRefFeatures
from core.dgp.utils.utils import image_rotate



class FlipTwoFeaturesV3(PatchRefFeatures):
    """Flip the image left & right"""
    def __init__(self, Z, patch_size, n_channel, name):
        gpflow.features.InducingFeature.__init__(self, name=name)

        self.input_dim = Z.shape[-1]
        self.patch_shape = [patch_size, patch_size, n_channel]

        self._setup_groups()
        self.Z1 = Parameter(Z, dtype=settings.float_type)
        self.Z2 = Parameter(Z, dtype=settings.float_type)

    def __len__(self):
        return self.Z1.shape[0] + self.Z2.shape[0]

    @params_as_tensors
    def save_inducing(self, sess, name):
        Z1, Z2 = sess.run([self.Z1, self.Z2])
        with open(name, 'wb') as file:
            np.savez(file, Z1=Z1, Z2=Z2)

    def _setup_groups(self):
        self.nT = 2
        self.DFT_matrix = tf.cast(np.array([[1, 1], [1, -1]]) / 2., settings.float_type)  # [2, 2]
        self.Ts = None

    @params_as_tensors
    def orbit(self, Z):
        Z_img = tf.reshape(Z, [tf.shape(Z)[0]] + self.patch_shape)
        Z_flip = tf.image.flip_left_right(Z_img)
        return [Z, tf.reshape(Z_flip, [tf.shape(Z)[0], np.prod(self.patch_shape)])]

    @name_scope('compute_PTMN_Kuf')
    @params_as_tensors
    def compute_PTMN_Kuf(self, NHWC_X, base_kern, view):  # minimal transpose operations
        N = tf.shape(NHWC_X)[0]
        output_size = view.out_image_height * view.out_image_width
        M = tf.shape(self.Z1)[0]

        # [batch_size, output_size_0, output_size_1, 1]
        # [num_inducing]
        # [batch_size, output_size_0, output_size_1, num_inducing]
        Z10, Z11 = self.orbit(self.Z1)
        Z20, Z21 = self.orbit(self.Z2)
        _, z1_square, patch_times_z10 = kerne_by_conv(NHWC_X, Z10, view, base_kern, return_patch_square=False)
        _, _, patch_times_z11 = kerne_by_conv(NHWC_X, Z11, view, base_kern, return_patch_square=False)
        _, _, patch_times_z20 = kerne_by_conv(NHWC_X, Z20, view, base_kern, return_patch_square=False)
        patch_square, z2_square, patch_times_z21 = kerne_by_conv(NHWC_X, Z21, view, base_kern)

        NPM_patch_square = tf.reshape(patch_square, [N, output_size, 1])
        NPM_patch_times_z10 = tf.reshape(patch_times_z10, [N, output_size, M])
        NPM_patch_times_z11 = tf.reshape(patch_times_z11, [N, output_size, M])
        NPM_patch_times_z20 = tf.reshape(patch_times_z20, [N, output_size, M])
        NPM_patch_times_z21 = tf.reshape(patch_times_z21, [N, output_size, M])

        PMN_patch_square = tf.transpose(NPM_patch_square, [1, 2, 0])
        PMN_patch_times_z10 = tf.transpose(NPM_patch_times_z10, [1, 2, 0])
        PMN_patch_times_z11 = tf.transpose(NPM_patch_times_z11, [1, 2, 0])
        PMN_patch_times_z20 = tf.transpose(NPM_patch_times_z20, [1, 2, 0])
        PMN_patch_times_z21 = tf.transpose(NPM_patch_times_z21, [1, 2, 0])


        PMN_square_sum_z1 = PMN_patch_square + z1_square[:, None]
        PMN_square_sum_z2 = PMN_patch_square + z2_square[:, None]

        PMN_Kuf_z10_plus = base_kern.K_r2(PMN_square_sum_z1 - 2 * PMN_patch_times_z10)
        PMN_Kuf_z11_nega = base_kern.K_r2(PMN_square_sum_z1 - 2 * PMN_patch_times_z11)

        PMN_Kuf_z20_plus = base_kern.K_r2(PMN_square_sum_z2 - 2 * PMN_patch_times_z20)
        PMN_Kuf_z21_nega = base_kern.K_r2(PMN_square_sum_z2 - 2 * PMN_patch_times_z21)

        PMN_Kuf_z1 = (PMN_Kuf_z10_plus + PMN_Kuf_z11_nega) / 2.
        PMN_Kuf_z2 = (PMN_Kuf_z20_plus - PMN_Kuf_z21_nega) / 2.

        PTMN_Kuf = tf.concat([PMN_Kuf_z1[:, None], PMN_Kuf_z2[:, None]], axis=1)
        return PTMN_Kuf


@dispatch(FlipTwoFeaturesV3, (MultiOutputConvKernel, ConvKernel))
def Kuu(feat, kern, *, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        Z1, Z2 = feat.Z1, feat.Z2
        norm_sq_Z1 = tf.reduce_sum(Z1 ** 2., -1)  # [M]
        norm_sq_Z2 = tf.reduce_sum(Z2 ** 2., -1)  # [M]

        Z10, Z11 = feat.orbit(Z1)
        inner_prod_Z1_Z10 = tf.matmul(Z1, Z10, transpose_b=True)
        inner_prod_Z1_Z11 = tf.matmul(Z1, Z11, transpose_b=True)
        Ks_z1_plus = kern.base_kern.from_inner_prod(norm_sq_Z1, inner_prod_Z1_Z10, norm_sq_Z1)
        Ks_z1_nega = kern.base_kern.from_inner_prod(norm_sq_Z1, inner_prod_Z1_Z11, norm_sq_Z1)

        Z20, Z21 = feat.orbit(Z2)
        inner_prod_Z2_Z20 = tf.matmul(Z2, Z20, transpose_b=True)
        inner_prod_Z2_Z21 = tf.matmul(Z2, Z21, transpose_b=True)
        Ks_z2_plus = kern.base_kern.from_inner_prod(norm_sq_Z2, inner_prod_Z2_Z20, norm_sq_Z2)
        Ks_z2_nega = kern.base_kern.from_inner_prod(norm_sq_Z2, inner_prod_Z2_Z21, norm_sq_Z2)

        Ks_z1 = (Ks_z1_plus + Ks_z1_nega) / 2.
        Ks_z2 = (Ks_z2_plus - Ks_z2_nega) / 2.

        Ks = tf.stack([Ks_z1, Ks_z2])
        Ks = Ks + jitter * tf.eye(tf.shape(Ks)[-1], dtype=Ks.dtype)  # [nT, M, M]
        return Ks


class FlipTwoLocFeaturesV3(FlipTwoFeaturesV3):

    def __init__(self, Z, patch_size, n_channel, name):
        super().__init__(Z[:, :-2], patch_size, n_channel, name)
        self.loc_z1 = Parameter(Z[:, -2:], dtype=settings.float_type)
        self.loc_z2 = Parameter(Z[:, -2:], dtype=settings.float_type)

    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return cls(features, patch_size, NHWC_X.shape[3], name)

@dispatch(FlipTwoLocFeaturesV3, TICKernel)
def Kuu(feat, kern, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        patch_kern, loc_kern = kern.patch_kern, kern.loc_kern

        Z1, Z2 = feat.Z1, feat.Z2
        norm_sq_Z1 = tf.reduce_sum(Z1 ** 2., -1)  # [M]
        norm_sq_Z2 = tf.reduce_sum(Z2 ** 2., -1)  # [M]

        Z10, Z11 = feat.orbit(Z1)
        inner_prod_Z1_Z10 = tf.matmul(Z1, Z10, transpose_b=True)
        inner_prod_Z1_Z11 = tf.matmul(Z1, Z11, transpose_b=True)
        Ks_z1_plus = patch_kern.from_inner_prod(norm_sq_Z1, inner_prod_Z1_Z10, norm_sq_Z1)
        Ks_z1_nega = patch_kern.from_inner_prod(norm_sq_Z1, inner_prod_Z1_Z11, norm_sq_Z1)

        Z20, Z21 = feat.orbit(Z2)
        inner_prod_Z2_Z20 = tf.matmul(Z2, Z20, transpose_b=True)
        inner_prod_Z2_Z21 = tf.matmul(Z2, Z21, transpose_b=True)
        Ks_z2_plus = patch_kern.from_inner_prod(norm_sq_Z2, inner_prod_Z2_Z20, norm_sq_Z2)
        Ks_z2_nega = patch_kern.from_inner_prod(norm_sq_Z2, inner_prod_Z2_Z21, norm_sq_Z2)

        Ks_z1 = (Ks_z1_plus + Ks_z1_nega) / 2.
        Ks_z2 = (Ks_z2_plus - Ks_z2_nega) / 2.

        Ks_z1 = Ks_z1 * loc_kern.K(feat.loc_z1, feat.loc_z1)
        Ks_z2 = Ks_z2 * loc_kern.K(feat.loc_z2, feat.loc_z2)

        Ks = tf.stack([Ks_z1, Ks_z2])

        return Ks + jitter * tf.eye(tf.shape(Ks)[-1], dtype=Ks.dtype)


@dispatch(FlipTwoLocFeaturesV3, TICKernel, object)
def Kuf(feature, kern, ND_X):
    with params_as_tensors_for(feature, kern):
        NHWC_X = kern._reshape_X(ND_X)
        locs = kern.get_loc()
        loc_Kzx = tf.stack([kern.loc_kern.K(feature.loc_z1, locs),
                            kern.loc_kern.K(feature.loc_z2, locs)]) # [nT, M, P]
        PTMN_Kuf = feature.compute_PTMN_Kuf(NHWC_X, kern.patch_kern, kern.view)
        PTMN_Kuf = PTMN_Kuf * tf.transpose(loc_Kzx, [2, 0, 1])[..., None]
        TMN_Kuf = tf.reduce_sum(PTMN_Kuf * kern.patch_weights[:, None, None, None], 0) / kern.patch_count
        return TMN_Kuf


class RotateTwoFeaturesV3(FlipTwoFeaturesV3):
    """rotate the image 180"""
    @params_as_tensors
    def orbit(self, Z):
        Z_img = tf.reshape(Z, [tf.shape(Z)[0]] + self.patch_shape)
        Z_flip = tf.image.flip_up_down(tf.image.flip_left_right(Z_img))
        return [Z, tf.reshape(Z_flip, [tf.shape(Z)[0], np.prod(self.patch_shape)])]

class RotateTwoLocFeaturesV3(FlipTwoLocFeaturesV3):
    @params_as_tensors
    def orbit(self, Z):
        Z_img = tf.reshape(Z, [tf.shape(Z)[0]] + self.patch_shape)
        Z_flip = tf.image.flip_up_down(tf.image.flip_left_right(Z_img))
        return [Z, tf.reshape(Z_flip, [tf.shape(Z)[0], np.prod(self.patch_shape)])]


class FlipUpdownTwoFeaturesV3(FlipTwoFeaturesV3):
    """flip the image up & down"""
    @params_as_tensors
    def orbit(self, Z):
        Z_img = tf.reshape(Z, [tf.shape(Z)[0]] + self.patch_shape)
        Z_flip = tf.image.flip_up_down(Z_img)
        return [Z, tf.reshape(Z_flip, [tf.shape(Z)[0], np.prod(self.patch_shape)])]


class FlipUpdownTwoLocFeaturesV3(FlipTwoLocFeaturesV3):
    """flip the image up & down"""
    @params_as_tensors
    def orbit(self, Z):
        Z_img = tf.reshape(Z, [tf.shape(Z)[0]] + self.patch_shape)
        Z_flip = tf.image.flip_up_down(Z_img)
        return [Z, tf.reshape(Z_flip, [tf.shape(Z)[0], np.prod(self.patch_shape)])]

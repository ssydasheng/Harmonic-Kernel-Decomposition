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


################## The Reflection PatchInducingFeatures -- Reflection by the mean ##################


class MirrorTwoFeaturesV3(PatchRefFeatures):
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
        self.Ts = tf.cast(np.vstack([np.ones([1, self.input_dim]),
                                     -np.ones([1, self.input_dim])]), settings.float_type)  # [2, d]

    @name_scope('compute_PTMN_Kuf')
    @params_as_tensors
    def compute_PTMN_Kuf(self, NHWC_X, base_kern, view):  # minimal transpose operations
        N = tf.shape(NHWC_X)[0]
        output_size = view.out_image_height * view.out_image_width
        M = tf.shape(self.Z1)[0]

        # [batch_size, output_size_0, output_size_1, 1]
        # [num_inducing]
        # [batch_size, output_size_0, output_size_1, num_inducing]
        _, z1_square, patch_times_z1 = kerne_by_conv(NHWC_X, self.Z1, view, base_kern, return_patch_square=False)
        patch_square, z2_square, patch_times_z2 = kerne_by_conv(NHWC_X, self.Z2, view, base_kern)

        NPM_patch_square = tf.reshape(patch_square, [N, output_size, 1])
        NPM_patch_times_z1 = tf.reshape(patch_times_z1, [N, output_size, M])
        NPM_patch_times_z2 = tf.reshape(patch_times_z2, [N, output_size, M])

        PMN_patch_square = tf.transpose(NPM_patch_square, [1, 2, 0])
        PMN_patch_times_z1 = tf.transpose(NPM_patch_times_z1, [1, 2, 0])
        PMN_patch_times_z2 = tf.transpose(NPM_patch_times_z2, [1, 2, 0])
        #
        # PTMN_patch_times_z1 = tf.concat([PMN_patch_times_z1[:, None], -PMN_patch_times_z1[:, None]], axis=1)
        # PTMN_patch_times_z2 = tf.concat([PMN_patch_times_z2[:, None], -PMN_patch_times_z2[:, None]], axis=1)
        #
        # PTMN_dist_z1 = PMN_patch_square[:, None] + z1_square[:, None] - 2 * PTMN_patch_times_z1
        # PTMN_dist_z2 = PMN_patch_square[:, None] + z2_square[:, None] - 2 * PTMN_patch_times_z2
        #
        # PTTMN_dist = tf.concat([PTMN_dist_z1[:, None], PTMN_dist_z2[:, None]], axis=1)
        # PTTMN_Kuf = base_kern.K_r2(PTTMN_dist)
        #
        # PTMN_Kuf = tf.reduce_sum(PTTMN_Kuf * self.DFT_matrix[..., None, None], 2)
        # return PTMN_Kuf

        PMN_square_sum_z1 = PMN_patch_square + z1_square[:, None]
        PMN_square_sum_z2 = PMN_patch_square + z2_square[:, None]

        PMN_Kuf_z1_plus = base_kern.K_r2(PMN_square_sum_z1 - 2 * PMN_patch_times_z1)
        PMN_Kuf_z1_nega = base_kern.K_r2(PMN_square_sum_z1 + 2 * PMN_patch_times_z1)

        PMN_Kuf_z2_plus = base_kern.K_r2(PMN_square_sum_z2 - 2 * PMN_patch_times_z2)
        PMN_Kuf_z2_nega = base_kern.K_r2(PMN_square_sum_z2 + 2 * PMN_patch_times_z2)

        PMN_Kuf_z1 = (PMN_Kuf_z1_plus + PMN_Kuf_z1_nega) / 2.
        PMN_Kuf_z2 = (PMN_Kuf_z2_plus - PMN_Kuf_z2_nega) / 2.

        PTMN_Kuf = tf.concat([PMN_Kuf_z1[:, None], PMN_Kuf_z2[:, None]], axis=1)
        return PTMN_Kuf


@dispatch(MirrorTwoFeaturesV3, (MultiOutputConvKernel, ConvKernel))
def Kuu(feat, kern, *, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        Z1, Z2 = feat.Z1, feat.Z2
        norm_sq_Z1 = tf.reduce_sum(Z1 ** 2., -1)  # [M]
        norm_sq_Z2 = tf.reduce_sum(Z2 ** 2., -1)  # [M]

        inner_prod_Z1 = tf.matmul(Z1, Z1, transpose_b=True)
        inner_prod_Z2 = tf.matmul(Z2, Z2, transpose_b=True)

        Ks_z1_plus = kern.base_kern.from_inner_prod(norm_sq_Z1, inner_prod_Z1, norm_sq_Z1)
        Ks_z1_nega = kern.base_kern.from_inner_prod(norm_sq_Z1, -inner_prod_Z1, norm_sq_Z1)

        Ks_z2_plus = kern.base_kern.from_inner_prod(norm_sq_Z2, inner_prod_Z2, norm_sq_Z2)
        Ks_z2_nega = kern.base_kern.from_inner_prod(norm_sq_Z2, -inner_prod_Z2, norm_sq_Z2)

        Ks_z1 = (Ks_z1_plus + Ks_z1_nega) / 2.
        Ks_z2 = (Ks_z2_plus - Ks_z2_nega) / 2.

        Ks = tf.stack([Ks_z1, Ks_z2])
        Ks = Ks + jitter * tf.eye(tf.shape(Ks)[-1], dtype=Ks.dtype)  # [nT, M, M]
        return Ks


class MirrorTwoLocFeaturesV3(MirrorTwoFeaturesV3):

    def __init__(self, Z, patch_size, n_channel, name):
        super().__init__(Z[:, :-2], patch_size, n_channel, name)
        self.loc_z1 = Parameter(Z[:, -2:], dtype=settings.float_type)
        self.loc_z2 = Parameter(Z[:, -2:], dtype=settings.float_type)

    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return cls(features, patch_size, NHWC_X.shape[3], name)

@dispatch(MirrorTwoLocFeaturesV3, TICKernel)
def Kuu(feat, kern, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        patch_kern, loc_kern = kern.patch_kern, kern.loc_kern

        Z1, Z2 = feat.Z1, feat.Z2
        norm_sq_Z1 = tf.reduce_sum(Z1 ** 2., -1)  # [M]
        norm_sq_Z2 = tf.reduce_sum(Z2 ** 2., -1)  # [M]

        inner_prod_Z1 = tf.matmul(Z1, Z1, transpose_b=True)
        inner_prod_Z2 = tf.matmul(Z2, Z2, transpose_b=True)

        Ks_z1_plus = patch_kern.from_inner_prod(norm_sq_Z1, inner_prod_Z1, norm_sq_Z1)
        Ks_z1_nega = patch_kern.from_inner_prod(norm_sq_Z1, -inner_prod_Z1, norm_sq_Z1)

        Ks_z2_plus = patch_kern.from_inner_prod(norm_sq_Z2, inner_prod_Z2, norm_sq_Z2)
        Ks_z2_nega = patch_kern.from_inner_prod(norm_sq_Z2, -inner_prod_Z2, norm_sq_Z2)

        Ks_z1 = (Ks_z1_plus + Ks_z1_nega) / 2.
        Ks_z2 = (Ks_z2_plus - Ks_z2_nega) / 2.

        Ks_z1 = Ks_z1 * loc_kern.K(feat.loc_z1, feat.loc_z1)
        Ks_z2 = Ks_z2 * loc_kern.K(feat.loc_z2, feat.loc_z2)

        Ks = tf.stack([Ks_z1, Ks_z2])

        return Ks + jitter * tf.eye(tf.shape(Ks)[-1], dtype=Ks.dtype)


@dispatch(MirrorTwoLocFeaturesV3, TICKernel, object)
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

import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, params_as_tensors, name_scope
from gpflow.params import Parameter
from gpflow import params_as_tensors_for
from gpflow.dispatch import dispatch

from core.dgp.utils.img_utils import _cluster_patches_with_locations, kerne_by_conv
from core.dgp.utils.kernels import MultiOutputConvKernel, ConvKernel, TICKernel

################## The Reflection PatchInducingFeatures ##################
class SinglePatchRefFeatures(gpflow.features.InducingFeature):
    def __init__(self, Z, j, patch_size, n_channel, name):
        super().__init__(name=name)

        self.input_dim = Z.shape[-1]
        self.patch_shape = [patch_size, patch_size, n_channel]

        self._setup_groups()
        self.j = j
        self.Z = Parameter(Z, dtype=settings.float_type)

    def _setup_groups(self):
        raise NotImplementedError

    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return cls, features[:, :-2]

    @params_as_tensors
    def save_inducing(self, sess, name):
        Z = sess.run(self.Z)
        with open(name, 'wb') as file:
            np.savez(file, Z=Z)

    @params_as_tensors
    def projective_inner_prod(self, Z1, Z2=None):
        if Z2 is None:
            Z2 = Z1
        return tf.matmul(Z1 * self.Ts[:, None, :], Z2, transpose_b=True)

    def __len__(self):
        return self.Z.shape[0]

    @params_as_tensors
    def compute_PMN_Kuf(self, NHWC_X, base_kern, view):
        raise NotImplementedError


class SingleMirrorFourFeatures(SinglePatchRefFeatures):

    def _setup_groups(self):
        patch_size, n_channel = self.patch_shape[0], self.patch_shape[-1]
        mask = np.mod(np.arange(patch_size)[:, None] + np.arange(patch_size), 2)
        self.mask = np.reshape(np.tile(mask[..., None], [1, 1, n_channel]), [-1])

        n_groups = 2
        T = 4

        # [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        # [T, n_groups]
        binary_matrix = (((np.arange(T)[:, None] & (1 << np.arange(n_groups)))) > 0).astype(float)
        binary_matrix = binary_matrix[:, ::-1]

        # [T, T]
        DFT_matrix = np.mod(binary_matrix @ binary_matrix.T, 2)
        self.DFT_matrix = tf.cast(np.where(DFT_matrix,
                                           -np.ones_like(DFT_matrix) / float(T),
                                           np.ones_like(DFT_matrix) / float(T)), settings.float_type)

        dimension_masks = []
        indicators = []
        for i in range(T):
            bin_mat = - 2 * binary_matrix[i] + 1
            mask = bin_mat[0] * self.mask + bin_mat[1] * (1-self.mask)
            dimension_masks.append(mask)
            indicators.append((bin_mat[0], bin_mat[1]))
        # [T, d]
        self.nT = T
        self.Ts = tf.cast(np.stack(dimension_masks), settings.float_type)
        self.indicators = indicators

    @name_scope('compute_PTMN_Kuf')
    @params_as_tensors
    def compute_PMN_Kuf(self, NHWC_X, base_kern, view):
        N = tf.shape(NHWC_X)[0]
        output_size = view.out_image_height * view.out_image_width
        M = tf.shape(self.Z)[0]

        Z1, Z2 = self.Z * self.mask, self.Z * (1 - self.mask)
        # [batch_size, output_size_0, output_size_1, 1]
        # [num_inducing]
        # [batch_size, output_size_0, output_size_1, num_inducing]
        _, z1_square, patch_times_z1 = kerne_by_conv(NHWC_X, Z1, view, base_kern, return_patch_square=False)
        patch_square, z2_square, patch_times_z2 = kerne_by_conv(NHWC_X, Z2, view, base_kern)
        z_square = z1_square + z2_square

        NPM_patch_square = tf.reshape(patch_square, [N, output_size, 1])
        NPM_patch_times_z1 = tf.reshape(patch_times_z1, [N, output_size, M])
        NPM_patch_times_z2 = tf.reshape(patch_times_z2, [N, output_size, M])

        PMN_patch_square = tf.transpose(NPM_patch_square, [1, 2, 0])
        PMN_patch_times_z1 = tf.transpose(NPM_patch_times_z1, [1, 2, 0])
        PMN_patch_times_z2 = tf.transpose(NPM_patch_times_z2, [1, 2, 0])

        PTMN_patch_times_zs = tf.concat(
            [PMN_patch_times_z1[:, None] * tf.cast(ind1, NHWC_X.dtype)
             + PMN_patch_times_z2[:, None] * tf.cast(ind2, NHWC_X.dtype)
             for ind1, ind2 in self.indicators],
            axis=1)

        PTMN_sq_dist_x_z = PMN_patch_square[:, None] + z_square[:, None] - 2 * PTMN_patch_times_zs
        PTMN_Kuf = base_kern.K_r2(PTMN_sq_dist_x_z)
        PMN_Kuf = tf.reduce_sum(PTMN_Kuf * self.DFT_matrix[:, self.j][..., None, None], 1)
        return PMN_Kuf

    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        def fn(j):
            def f(z):
                return cls(z, j, patch_size, NHWC_X.shape[3], name+'_%d' % j)
            return f
        return [(fn(j), features[:, :-2]) for j in range(4)]

@dispatch(SinglePatchRefFeatures, (MultiOutputConvKernel, ConvKernel))
def Kuu(feat, kern, *, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        norm_sq_Zs = tf.reduce_sum(feat.Z**2., -1) # [M]
        inner_prod_Zs = feat.projective_inner_prod(feat.Z) # [nT, M, M]
        Ks = kern.base_kern.from_inner_prod(norm_sq_Zs, inner_prod_Zs, norm_sq_Zs) # [nT, M, M]
        Ks = tf.reduce_sum(Ks * feat.DFT_matrix[:, feat.j][:, None, None], 0) # [M, M]
        Ks = Ks + jitter*tf.eye(tf.shape(Ks)[-1], dtype=Ks.dtype) # [M, M]
        return Ks


@dispatch(SinglePatchRefFeatures, MultiOutputConvKernel, object)
def Kuf(feat, kern, NHWC_X):
    return feat.compute_PMN_Kuf(NHWC_X, kern.base_kern, kern.view)


@dispatch(SinglePatchRefFeatures, ConvKernel, object)
def Kuf(feat, kern, ND_X):
    with params_as_tensors_for(kern):
        NHWC_X = kern._reshape_X(ND_X)
        PMN_Kuf = feat.compute_PMN_Kuf(NHWC_X, kern.base_kern, kern.view)

        MN_Kuf = tf.reduce_sum(PMN_Kuf * kern.patch_weights[:, None, None], 0) / kern.patch_count
        return MN_Kuf



################## The TICK PatchReflectionFeatures, with Locations ##################

class SinglePatchLocRefFeatures(SinglePatchRefFeatures):
    def __init__(self, Z, j, patch_size, n_channel, name):
        super().__init__(Z[:, :-2], j, patch_size, n_channel, name)
        self.loc_z = Parameter(Z[:, -2:], dtype=settings.float_type)


class SingleMirrorFourLocFeatures(SinglePatchLocRefFeatures, SingleMirrorFourFeatures):
    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        def fn(j):
            def f(z):
                return cls(z, j, patch_size, NHWC_X.shape[3], name+'_%d' % j)
            return f
        return [(fn(j), features) for j in range(4)]


@dispatch(SinglePatchLocRefFeatures, TICKernel)
def Kuu(feat, kern, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        patch_kern, loc_kern = kern.patch_kern, kern.loc_kern

        norm_sq_Zs = tf.reduce_sum(feat.Z**2., -1) # [M]
        inner_prod_Zs = feat.projective_inner_prod(feat.Z) # [nT, M, M]
        Ks = patch_kern.from_inner_prod(norm_sq_Zs, inner_prod_Zs, norm_sq_Zs) # [nT, M, M]
        patch_Ks = tf.reduce_sum(Ks * feat.DFT_matrix[:, feat.j][:, None, None], 0)  # [M, M]

        res = patch_Ks * loc_kern.K(feat.loc_z, feat.loc_z)
        return res +jitter*tf.eye(tf.shape(patch_Ks)[-1], dtype=Ks.dtype) # [M, M]


@dispatch(SinglePatchLocRefFeatures, TICKernel, object)
def Kuf(feature, kern, ND_X):
    with params_as_tensors_for(feature, kern):
        NHWC_X = kern._reshape_X(ND_X)
        locs = kern.get_loc()
        loc_Kzx = kern.loc_kern.K(feature.loc_z, locs)

        PMN_Kuf = feature.compute_PMN_Kuf(NHWC_X, kern.patch_kern, kern.view)
        PMN_Kuf = PMN_Kuf * tf.transpose(loc_Kzx)[:, :, None]
        MN_Kuf = tf.reduce_sum(PMN_Kuf * kern.patch_weights[:, None, None], 0) / kern.patch_count
        return MN_Kuf

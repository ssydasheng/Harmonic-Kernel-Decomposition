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


################## The standard PatchInducingFeatures ##################

class PatchInducingFeatures(InducingPoints):
    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return PatchInducingFeatures(features[:,:-2], name=name)


@dispatch(PatchInducingFeatures, (MultiOutputConvKernel, ConvKernel))
def Kuu(feature, kern, jitter=0.0):
    M = len(feature)
    # [M, M]
    return gpflow.features.Kuu(feature, kern.base_kern) + tf.eye(M, dtype=settings.float_type) * jitter

@dispatch(PatchInducingFeatures, MultiOutputConvKernel, object)
def Kuf(feature, kern, NHWC_X):
    with params_as_tensors_for(feature):
        NPM_Kuf = fast_kernel_by_conv(NHWC_X, feature.Z, kern.view, kern.base_kern)
        PMN_Kuf = tf.transpose(NPM_Kuf, [1, 2, 0])
        return PMN_Kuf

@dispatch(PatchInducingFeatures, ConvKernel, object)
def Kuf(feature, kern, ND_X):
    with params_as_tensors_for(feature):
        NHWC_X = kern._reshape_X(ND_X)
        assert isinstance(kern.base_kern, Stationary)
        NPM_K = fast_kernel_by_conv(NHWC_X, feature.Z, kern.view, kern.base_kern)
        Kzx = tf.transpose(NPM_K, [2, 0, 1])
        w = kern.patch_weights
        Kzx = Kzx * w
        Kzx = tf.reduce_sum(Kzx, [2])
        return Kzx / kern.patch_count


################## The TICK PatchInducingFeatures, with Locations ##################

class PatchLocInducingFeatures(InducingPoints):
    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return PatchLocInducingFeatures(features, name=name)

@dispatch(PatchLocInducingFeatures, TICKernel)
def Kuu(feature, kern, jitter=0.0):
    with params_as_tensors_for(feature):
        patch_kern, loc_kern = kern.patch_kern, kern.loc_kern
        Z = feature.Z
        res = patch_kern.K(Z[:, :-2], Z[:, :-2]) * loc_kern.K(Z[:, -2:], Z[:, -2:])
        # [M, M]
        return res + tf.eye(len(feature), dtype=settings.float_type) * jitter

@dispatch(PatchLocInducingFeatures, TICKernel, object)
def Kuf(feature, kern, ND_X):
    with params_as_tensors_for(feature, kern):
        Z = feature.Z
        NHWC_X = kern._reshape_X(ND_X)
        locs = kern.get_loc()
        loc_Kzx = kern.loc_kern.K(Z[:, -2:], locs)

        assert isinstance(kern.patch_kern, Stationary)
        NPM_K = fast_kernel_by_conv(NHWC_X, Z[:, :-2], kern.view, kern.patch_kern)
        patch_Kzx = tf.transpose(NPM_K, [2, 0, 1])
        Kzx = patch_Kzx * loc_Kzx[:, None]
        Kzx = Kzx * kern.patch_weights
        Kzx = tf.reduce_sum(Kzx, [2])
        return Kzx / kern.patch_count


################## The Reflection PatchInducingFeatures ##################
class PatchRefFeatures(gpflow.features.InducingFeature):
    def __init__(self, Z, patch_size, n_channel, name):
        super().__init__(name=name)

        self.input_dim = Z.shape[-1]
        self.patch_shape = [patch_size, patch_size, n_channel]

        self._setup_groups()
        self.Z = Parameter(Z, dtype=settings.float_type)

    def _setup_groups(self):
        raise NotImplementedError

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

    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return cls(features[:,:-2], patch_size, NHWC_X.shape[3], name)

    def __len__(self):
        return self.nT * self.Z.shape[0]

    @params_as_tensors
    def compute_PTMN_Kuf(self, NHWC_X, base_kern, view):
        raise NotImplementedError


class MirrorTwoFeatures(PatchRefFeatures):
    def _setup_groups(self):
        self.nT = 2
        self.DFT_matrix = tf.cast(np.array([[1, 1], [1, -1]]) / 2., settings.float_type) # [2, 2]
        self.Ts = tf.cast(np.vstack([np.ones([1, self.input_dim]),
                                     -np.ones([1, self.input_dim])]), settings.float_type) # [2, d]

    @name_scope('compute_PTMN_Kuf')
    @params_as_tensors
    def compute_PTMN_Kuf(self, NHWC_X, base_kern, view): # minimal transpose operations
        N = tf.shape(NHWC_X)[0]
        output_size = view.out_image_height * view.out_image_width
        M = tf.shape(self.Z)[0]

        # [batch_size, output_size_0, output_size_1, 1]
        # [num_inducing]
        # [batch_size, output_size_0, output_size_1, num_inducing]
        patch_square, z_square, patch_times_z = kerne_by_conv(NHWC_X, self.Z, view, base_kern)

        NPM_patch_square = tf.reshape(patch_square, [N, output_size, 1])
        NPM_patch_times_z = tf.reshape(patch_times_z, [N, output_size, M])

        PMN_patch_square = tf.transpose(NPM_patch_square, [1, 2, 0])
        PMN_patch_times_z = tf.transpose(NPM_patch_times_z, [1, 2, 0])

        PTMN_patch_times_zs = tf.concat([PMN_patch_times_z[:, None], -PMN_patch_times_z[:, None]], axis=1)

        PTMN_sq_dist_x_z = PMN_patch_square[:, None] + z_square[:, None] - 2 * PTMN_patch_times_zs
        PTMN_Kuf = base_kern.K_r2(PTMN_sq_dist_x_z)

        PTMN_Kuf = tf.reduce_sum(PTMN_Kuf[:, None] * self.DFT_matrix[..., None, None], 2)
        return PTMN_Kuf

class MirrorFourFeatures(PatchRefFeatures):

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
    def compute_PTMN_Kuf(self, NHWC_X, base_kern, view):
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
        PTMN_Kuf = tf.reduce_sum(PTMN_Kuf[:, None] * self.DFT_matrix[..., None, None], 2)
        return PTMN_Kuf


@dispatch(PatchRefFeatures, (MultiOutputConvKernel, ConvKernel))
def Kuu(feat, kern, *, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        norm_sq_Zs = tf.reduce_sum(feat.Z**2., -1) # [M]
        inner_prod_Zs = feat.projective_inner_prod(feat.Z) # [nT, M, M]
        Ks = kern.base_kern.from_inner_prod(norm_sq_Zs, inner_prod_Zs, norm_sq_Zs) # [nT, M, M]

        Ks = tf.matmul(tf.transpose(Ks, [1, 2, 0]), feat.DFT_matrix, transpose_b=True) # [M, M, nT]
        Ks = tf.transpose(Ks, [2, 0, 1]) + jitter*tf.eye(tf.shape(Ks)[0], dtype=Ks.dtype) # [nT, M, M]
        return Ks


@dispatch(PatchRefFeatures, MultiOutputConvKernel, object)
def Kuf(feat, kern, NHWC_X):
    return feat.compute_PTMN_Kuf(NHWC_X, kern.base_kern, kern.view)


@dispatch(PatchRefFeatures, ConvKernel, object)
def Kuf(feat, kern, ND_X):
    with params_as_tensors_for(kern):
        NHWC_X = kern._reshape_X(ND_X)
        PTMN_Kuf = feat.compute_PTMN_Kuf(NHWC_X, kern.base_kern, kern.view)

        TMN_Kuf = tf.reduce_sum(PTMN_Kuf * kern.patch_weights[:, None, None, None], 0) / kern.patch_count
        return TMN_Kuf




################## The TICK PatchReflectionFeatures, with Locations ##################

class PatchLocRefFeatures(PatchRefFeatures):
    def __init__(self, Z, patch_size, n_channel, name):
        super().__init__(Z[:, :-2], patch_size, n_channel, name)
        self.loc_z = Parameter(Z[:, -2:], dtype=settings.float_type)

    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return cls(features, patch_size, NHWC_X.shape[3], name)


class MirrorTwoLocFeatures(PatchLocRefFeatures, MirrorTwoFeatures):
    pass

class MirrorFourLocFeatures(PatchLocRefFeatures, MirrorFourFeatures):
    pass

@dispatch(PatchLocRefFeatures, TICKernel)
def Kuu(feat, kern, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        patch_kern, loc_kern = kern.patch_kern, kern.loc_kern

        norm_sq_Zs = tf.reduce_sum(feat.Z**2., -1) # [M]
        inner_prod_Zs = feat.projective_inner_prod(feat.Z) # [nT, M, M]
        Ks = patch_kern.from_inner_prod(norm_sq_Zs, inner_prod_Zs, norm_sq_Zs) # [nT, M, M]

        Ks = tf.matmul(tf.transpose(Ks, [1, 2, 0]), feat.DFT_matrix, transpose_b=True) # [M, M, nT]
        patch_Ks = tf.transpose(Ks, [2, 0, 1])  # [nT, M, M]

        res = patch_Ks * loc_kern.K(feat.loc_z, feat.loc_z)
        return res +jitter*tf.eye(tf.shape(patch_Ks)[-1], dtype=Ks.dtype)

@dispatch(PatchLocRefFeatures, TICKernel, object)
def Kuf(feature, kern, ND_X):
    with params_as_tensors_for(feature, kern):
        NHWC_X = kern._reshape_X(ND_X)
        locs = kern.get_loc()
        loc_Kzx = kern.loc_kern.K(feature.loc_z, locs)

        PTMN_Kuf = feature.compute_PTMN_Kuf(NHWC_X, kern.patch_kern, kern.view)
        PTMN_Kuf = PTMN_Kuf * tf.transpose(loc_Kzx)[:, None, :, None]
        TMN_Kuf = tf.reduce_sum(PTMN_Kuf * kern.patch_weights[:, None, None, None], 0) / kern.patch_count
        return TMN_Kuf



################## The TICK PatchReflectionFeatures, use different Locations ##################

class PatchLocRefFeaturesV2(PatchRefFeatures):
    def __init__(self, Z, patch_size, n_channel, name):
        super().__init__(Z[:, :-2], patch_size, n_channel, name)
        self.setup_loc_zs(Z[:, -2:])

    def setup_loc_zs(self, Z):
        raise NotImplementedError

    @classmethod
    def from_images(cls, NHWC_X, M, patch_size, name):
        features = _cluster_patches_with_locations(NHWC_X, M, patch_size)
        return cls(features, patch_size, NHWC_X.shape[3], name)



class MirrorTwoLocFeaturesV2(PatchLocRefFeaturesV2, MirrorTwoFeatures):
    def setup_loc_zs(self, Z):
        self.loc_zs = ParamList([
            Parameter(Z, dtype=settings.float_type),
            Parameter(Z, dtype=settings.float_type)])

class MirrorFourLocFeaturesV2(PatchLocRefFeaturesV2, MirrorFourFeatures):
    def setup_loc_zs(self, Z):
        self.loc_zs = ParamList([
            Parameter(Z, dtype=settings.float_type),
            Parameter(Z, dtype=settings.float_type),
            Parameter(Z, dtype=settings.float_type),
            Parameter(Z, dtype=settings.float_type)])

@dispatch(PatchLocRefFeaturesV2, TICKernel)
def Kuu(feat, kern, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        patch_kern, loc_kern = kern.patch_kern, kern.loc_kern

        norm_sq_Zs = tf.reduce_sum(feat.Z**2., -1) # [M]
        inner_prod_Zs = feat.projective_inner_prod(feat.Z) # [nT, M, M]
        Ks = patch_kern.from_inner_prod(norm_sq_Zs, inner_prod_Zs, norm_sq_Zs) # [nT, M, M]

        Ks = tf.matmul(tf.transpose(Ks, [1, 2, 0]), feat.DFT_matrix, transpose_b=True) # [M, M, nT]
        patch_Ks = tf.transpose(Ks, [2, 0, 1])  # [nT, M, M]

        loc_Ks = tf.stack([loc_kern.K(loc_z, loc_z) for loc_z in feat.loc_zs]) # [nT, M, M]

        res = patch_Ks * loc_Ks
        return res +jitter*tf.eye(tf.shape(patch_Ks)[-1], dtype=Ks.dtype)

@dispatch(PatchLocRefFeaturesV2, TICKernel, object)
def Kuf(feature, kern, ND_X):
    with params_as_tensors_for(feature, kern):
        NHWC_X = kern._reshape_X(ND_X)
        locs = kern.get_loc()
        loc_Kzx = tf.stack([kern.loc_kern.K(loc_z, locs) for loc_z in feature.loc_zs]) # [nT, M, P]

        PTMN_Kuf = feature.compute_PTMN_Kuf(NHWC_X, kern.patch_kern, kern.view)
        PTMN_Kuf = PTMN_Kuf * tf.transpose(loc_Kzx, [2, 0, 1])[..., None]
        TMN_Kuf = tf.reduce_sum(PTMN_Kuf * kern.patch_weights[:, None, None, None], 0) / kern.patch_count
        return TMN_Kuf
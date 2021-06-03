import numpy as np
import tensorflow as tf
import gpflow


def DFT_REAL_MATRIX(J):
    if J == 2:
        return np.asarray([[1., 1.], [1., -1.]]) / 2.
    JJt = np.arange(J).reshape([J, 1]) * np.arange(J).reshape([1, J])
    real = np.cos(2 * np.pi / J * JJt) / J
    return real

####################################################################################

class VaryInducingFeatures(gpflow.features.InducingFeature):
    """Vary the Inducing Features for all Groups
    It needs the attribute self.Z and self.Z_orbit.
    The self.Z is a list of length T whose element is a tensor of shape [M, d].
    The self.Z_orbit is a list of length T whose element is a tensor of shape [T, M, d].
    It also needs the attribute self.DFT_matrix of shape [T, T].
    """
    def __len__(self):
        with gpflow.params_as_tensors_for(self):
            if isinstance(self.Zs[0], tf.Tensor):
                return tf.add_n([Z.get_shape().as_list()[0] for Z in self.Zs])
            return sum([Z.shape[0] for Z in self.Zs])

    def shape(self, j):
        return self.Zs[j].shape[0]


@gpflow.dispatch.dispatch(VaryInducingFeatures, gpflow.kernels.Kernel)
def Kuu(feat, kern, *, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        Ks = [kern.K(Z, rZ) for Z, rZ in zip(feat.Zs, feat.Z_orbit)] # list of [nT, M, M]
        DFT = feat.DFT_matrix # [T, T]
        Ks = [tf.reduce_sum(tf.cast(K, DFT.dtype) * DFT[j][:, None], -2) for j, K in enumerate(Ks)]
        Ks = [K + jitter*tf.eye(tf.shape(K)[-1], dtype=DFT.dtype) for j, K in enumerate(Ks)]
        Ks = tf.stack(Ks)
        return Ks


@gpflow.dispatch.dispatch(VaryInducingFeatures, gpflow.kernels.Kernel, object)
def Kuf(feat, kern, Xnew):
    with gpflow.params_as_tensors_for(feat):
        Ks = [kern.K(Xnew, rZ) for rZ in feat.Z_orbit]  # list of [M, nT, M]
        DFT = feat.DFT_matrix  # [T, T]
        Ks = [tf.transpose(tf.reduce_sum(tf.cast(K, DFT.dtype) * DFT[j][:, None], -2))
              for j, K in enumerate(Ks)]
        Ks = tf.stack(Ks)
        return Ks

class ShareInducingFeatures(gpflow.features.InducingFeature):
    """Share the Inducing Features for all Groups
    It needs the attribute self.Z and self.Z_orbit,
    The self.Z is a tensor of shape [M, d].
    The self.Z_orbit is a tensor of shape [T, M, d].
    It also needs the attribute self.DFT_matrix of shape [T, T].
    """
    def __len__(self):
        if isinstance(self.Z, tf.Tensor):
            return self.Z.get_shape().as_list()[0] * self.num_groups
        return self.Z.shape[0] * self.num_groups

    def shape(self, j):
        return self.Z.shape[0]

@gpflow.dispatch.dispatch(ShareInducingFeatures, gpflow.kernels.Kernel)
def Kuu(feat, kern, *, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        Ks = kern.K(feat.Z, feat.Z_orbit) # [M, nT, M]
        DFT = feat.DFT_matrix # [T, T]
        Ks = tf.transpose(tf.matmul(DFT, Ks), [1, 0, 2])
        Ks = Ks + jitter * tf.eye(tf.shape(Ks)[-1], dtype=DFT.dtype)
        return Ks

@gpflow.dispatch.dispatch(ShareInducingFeatures, gpflow.kernels.Kernel, object)
def Kuf(feat, kern, Xnew):
    with gpflow.params_as_tensors_for(feat):
        Ks = kern.K(Xnew, feat.Z_orbit)  # [N, nT, M]
        DFT = feat.DFT_matrix  # [T, T]
        Ks = tf.transpose(tf.matmul(DFT, Ks), [1, 2, 0])
        return Ks


class SingleInducingPoints(gpflow.features.InducingFeature):
    """Inducing points for one single group.
    It needs the attribute self.Z and self.Z_orbit,
    The self.Z is a tensor of shape [M, d].
    The self.Z_orbit is a tensor of shape [T, M, d].
    It also needs the attribute self.DFT_matrix of shape [T, T].
    """
    def __len__(self):
        if isinstance(self.Z, tf.Tensor):
            return self.Z.get_shape().as_list()[0]
        return self.Z.shape[0]

@gpflow.dispatch.dispatch(SingleInducingPoints, gpflow.kernels.Kernel)
def Kuu(feat, kern, *, jitter=0.0):
    with gpflow.params_as_tensors_for(feat, kern):
        Ks = kern.K(feat.Z, feat.Z_orbit) # list of [nT, M, M]
        DFT = feat.DFT_matrix # [T, T]
        Ks = tf.reduce_sum(tf.cast(Ks, DFT.dtype) * DFT[feat.j][:, None], -2)
        Ks = Ks + jitter*tf.eye(tf.shape(Ks)[-1], dtype=DFT.dtype)
        return Ks

@gpflow.dispatch.dispatch(SingleInducingPoints, gpflow.kernels.Kernel, object)
def Kuf(feat, kern, Xnew):
    with gpflow.params_as_tensors_for(feat):
        Ks = kern.K(Xnew, feat.Z_orbit)  # list of [nT, M, M]
        DFT = feat.DFT_matrix  # [T, T]
        Ks = tf.transpose(tf.reduce_sum(tf.cast(Ks, DFT.dtype) * DFT[feat.j][:, None], -2))
        return Ks

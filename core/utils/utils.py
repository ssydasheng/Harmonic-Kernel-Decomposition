import numpy as np

import gpflow
from gpflow import settings, kernels
from gpflow.params import Parameter
from gpflow.dispatch import dispatch
from gpflow.decors import params_as_tensors_for, params_as_tensors
import tensorflow as tf


def block_diagonal(A, B):
    nA, nB = tf.shape(A)[0], tf.shape(B)[0]
    dtype = A.dtype

    A_zero = tf.concat([A, tf.zeros([nA, nB], dtype=dtype)], 1)
    zero_B = tf.concat([tf.zeros([nB, nA], dtype=dtype), B], 1)
    AB = tf.concat([A_zero, zero_B], 0)
    return AB


def triangular_solve_block_diagonal(A, B, n, lower=True):
    # compute A^{-1} B
    # A: "lower" triangular 2x2 block diagonal of shape [..., N, N]
    # B: shape [..., N, M]

    A1 = A[..., :n, :n]
    B1 = B[..., :n, :]
    C1 = tf.matrix_triangular_solve(A1, B1, lower=lower) # [..., n, M]

    A2 = A[..., n:, n:]
    B2 = B[..., n:, :]
    C2 = tf.matrix_triangular_solve(A2, B2, lower=lower) # [..., N-n, M]

    C = tf.concat([C1, C2], axis=-2)  # [..., N, M]
    return C


def matmul_block_diagonal(A, B, n, transpose_a=False):
    # compute A B
    # A: 2x2 block diagonal of shape [..., N, N]
    # B: shape [..., N, M]
    # transpose_a: whether transpose_a

    A1 = A[..., :n, :n]
    B1 = B[..., :n, :]
    C1 = tf.matmul(A1, B1, transpose_a=transpose_a)

    A2 = A[..., n:, n:]
    B2 = B[..., n:, :]
    C2 = tf.matmul(A2, B2, transpose_a=transpose_a)

    C = tf.concat([C1, C2], axis=-2) # [..., N, M]
    return C




class InducingPointsBase(gpflow.features.InducingFeature):
    """
    Real-space inducing points
    """

    def __init__(self, Z, name):
        """
        :param Z: the initial positions of the inducing points, size M x D
        """
        super().__init__(name=name)
        self.Z = Parameter(Z, dtype=settings.float_type)

    def __len__(self):
        return self.Z.shape[0]

    @params_as_tensors
    def save_inducing(self, sess, name):
        Z = sess.run(self.Z)
        with open(name, 'wb') as file:
            np.savez(file, Z=Z)


class InducingPoints(InducingPointsBase):
    pass

@dispatch(InducingPoints, kernels.Kernel)
def Kuu(feat, kern, *, jitter=0.0):
    with params_as_tensors_for(feat):
        Kzz = kern.K(feat.Z)
        Kzz += jitter * tf.eye(len(feat), dtype=settings.float_type)
    return Kzz


@dispatch(InducingPoints, kernels.Kernel, object)
def Kuf(feat, kern, Xnew):
    with params_as_tensors_for(feat):
        Kzx = kern.K(feat.Z, Xnew)
    return Kzx


class BatchLowerTriangular(gpflow.transforms.Transform):
    def __init__(self, N):
        self.N = N
        self.mask = np.tril(np.ones([self.N, self.N]))

    def forward(self, x):
        return x * self.mask

    def backward(self, y):
        return y * self.mask

    def forward_tensor(self, x):
        return x * tf.cast(self.mask, x.dtype)

    def backward_tensor(self, y):
        return y * tf.cast(self.mask, y.dtype)

    def log_jacobian_tensor(self, x):
        raise NotImplementedError

    def __str__(self):
        return "LoTriMat"


def none_sum(*args):
    subset = [a for a in args if a is not None]
    if len(subset) == 0:
        return None
    else:
        return tf.add_n(subset)
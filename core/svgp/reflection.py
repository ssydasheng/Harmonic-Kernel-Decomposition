import numpy as np
import tensorflow as tf
from gpflow import settings
from gpflow.params import Parameter, ParamList

from core.svgp.features import VaryInducingFeatures, SingleInducingPoints


def setup_groups(input_dim, T):
    n_groups = int(np.log2(T))
    assert T == np.power(2, n_groups)

    # [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    # [T, n_groups]
    binary_matrix = (((np.arange(T)[:, None] & (1 << np.arange(n_groups)))) > 0).astype(float)
    binary_matrix = binary_matrix[:, ::-1]

    # [T, T]
    DFT_matrix = np.mod(binary_matrix @ binary_matrix.T, 2)
    DFT_matrix = np.where(DFT_matrix,
                          -np.ones_like(DFT_matrix) / float(T),
                          np.ones_like(DFT_matrix) / float(T))

    flags = []
    for id in range(n_groups):
        zeros = np.zeros([input_dim])
        set = np.arange(id, input_dim, n_groups)
        zeros[set] = np.ones([len(set)])
        flags.append(zeros)
    dimension_masks = []
    for i in range(T):
        bin_mat = - 2 * binary_matrix[i] + 1
        mask = sum([bin_mat[j] * f for j, f in enumerate(flags)])
        dimension_masks.append(mask)
    # [T, d]
    masks = np.stack(dimension_masks)
    return DFT_matrix, masks


####################################################################################
class ReflectionFeatures(VaryInducingFeatures):
    """
    The HVGP feature whose transformation is negating the value along PCA directions.

    Sigma_x = U Lambda Ut, thus U captures the primary directions of input domain.
    We conduct transformation on the eigen directions of U.

    \tilde{z} = U^t z ==> \tilde{z}_{mask} = \tilde{z} \circ mask
                      ==> z_{mask} = U \tilde{z}_{mask}
                                   = U ((U^t z ) \circ mask)
    """
    def __init__(self, zs, U, name):
        super().__init__(name=name)
        self.num_groups = len(zs)
        self._setup_groups(zs[0].shape[-1], len(zs))
        self.U = tf.cast(U, dtype=settings.float_type)
        self.Zs = ParamList([Parameter(z, dtype=settings.float_type, name='Z%d'%idx)
                             for idx, z in enumerate(zs)])
        self.Zs.build()
        self.Z_orbit = [self.orbit(Z.constrained_tensor) for Z in self.Zs]

    def _setup_groups(self, input_dim, T):
        DFT_matrix, masks = setup_groups(input_dim, T)
        self.DFT_matrix = tf.constant(DFT_matrix, dtype=settings.float_type)
        self.masks = tf.constant(masks, dtype=settings.float_type)

    def orbit(self, Z):
        tilde_z_mask = tf.matmul(Z, self.U) * self.masks[:, None, :] # [T, M, D]
        z_mask = tf.matmul(tilde_z_mask, self.U, transpose_b=True)
        return z_mask



################# This used for multi-gpu training, list represents different GPU groups. ##########
class ReflectionSingleFeatures(SingleInducingPoints):
    def __init__(self, z, U, j, T, name):
        super().__init__(name=name)
        self.num_groups = T
        self._setup_groups(z.shape[-1], T)
        self.Z = Parameter(z, dtype=settings.float_type, name=name)
        self.j = j
        self.U = tf.cast(U, dtype=settings.float_type)

        self.Z.build()
        self.Z_orbit = self.orbit(self.Z.constrained_tensor)

    def _setup_groups(self, input_dim, T):
        DFT_matrix, masks = setup_groups(input_dim, T)
        self.DFT_matrix = tf.constant(DFT_matrix, dtype=settings.float_type)
        self.masks = tf.constant(masks, dtype=settings.float_type)

    def orbit(self, Z):
        tilde_z_mask = tf.matmul(Z, self.U) * self.masks[:, None, :]  # [T, M, D]
        z_mask = tf.matmul(tilde_z_mask, self.U, transpose_b=True) # [T, M, D]
        return z_mask

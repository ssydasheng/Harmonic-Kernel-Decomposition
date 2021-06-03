import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.params import Parameter, ParamList
from gpflow import params_as_tensors

from core.svgp.features import VaryInducingFeatures, ShareInducingFeatures, DFT_REAL_MATRIX


class AbstractImageFeatures(VaryInducingFeatures):
    def __init__(self, zs, patch_shape, name='imagerotation'):
        super().__init__(name=name)
        self.num_groups = len(zs)
        self.Zs = ParamList([Parameter(z, dtype=settings.float_type) for z in zs])
        self.patch_shape = patch_shape
        self._setup_DFT_matrix()
        self.Zs.build()
        self.Z_orbit = [self.orbit(Z.constrained_tensor) for Z in self.Zs]

    def _setup_DFT_matrix(self):
        raise NotImplementedError

    @params_as_tensors
    def orbit(self, Z):
        raise NotImplementedError


class ImageFlipFeatures(AbstractImageFeatures):
    """ HVGPs with the Flipping transformation """
    def __init__(self, zs, patch_shape, name='imageflip'):
        super().__init__(zs, patch_shape, name=name)

    def _setup_DFT_matrix(self):
        DFT2 = np.array([[1., 1.], [1., -1.]]) / 2.
        DFT4 = DFT2[:, None, :, None] * DFT2[:, None, :]
        self.DFT_matrix = tf.cast(DFT4.reshape([4, 4]), settings.float_type)

    @params_as_tensors
    def orbit(self, Z):
        Z = tf.reshape(Z, [tf.shape(Z)[0]] + self.patch_shape)
        Z1 = tf.image.flip_left_right(Z)
        Z2 = tf.image.flip_up_down(Z)
        Z3 = tf.image.flip_up_down(Z1)
        return tf.reshape(tf.stack([Z, Z1, Z2, Z3]), [4, tf.shape(Z)[0], np.prod(self.patch_shape)])


class ImageTranslateFeatures(AbstractImageFeatures):
    """ HVGPs with the Translating transformation """
    def __init__(self, zs, step, num_translate, patch_shape, name='imagetranslation'):
        gpflow.features.InducingFeature.__init__(self, name=name)
        self.Zs = ParamList([Parameter(z, dtype=settings.float_type) for z in zs])
        self.patch_shape = patch_shape

        self.num_translate = num_translate
        self.step = step
        self.groups_per_direction = int(np.floor(num_translate / 2.) + 1)
        self.num_groups = self.groups_per_direction ** 2
        assert self.num_groups == len(zs)

        self._setup_DFT_matrix()
        self.Zs.build()
        self.Z_orbit = [self.orbit(Z.constrained_tensor) for Z in self.Zs]

    @params_as_tensors
    def orbit(self, Z):
        Z = tf.reshape(Z, [tf.shape(Z)[0]] + self.patch_shape)

        # [num_translate, M, H, H, C]
        Z_cols = tf.stack([tf.roll(Z, j * self.step, axis=-2) for j in range(self.num_translate)])
        # [num_translate, num_translate, M, H, H, C]
        Z_rows_cols = tf.stack([tf.roll(Z_cols, j*self.step, axis=-3) for j in range(self.num_translate)])
        return tf.reshape(Z_rows_cols, [self.num_translate**2, tf.shape(Z)[0], np.prod(self.patch_shape)])

    def _setup_DFT_matrix(self):
        REAL = DFT_REAL_MATRIX(self.num_translate)
        RES = 2*REAL
        RES[0] = REAL[0]
        if self.num_translate % 2 == 0:
            RES[self.groups_per_direction-1] = REAL[self.groups_per_direction-1]

        DFT_row = RES[:self.groups_per_direction] # [groups_per_direction, num_translate]
        # [groups_per_direction, groups_per_direction, num_translate, num_translate]
        DFT_matrix = DFT_row[:, None, :, None] * DFT_row[:, None, :]
        self.DFT_matrix = tf.reshape(DFT_matrix, [self.groups_per_direction**2, self.num_translate**2])


class ShareImageTranslateFeatures(ShareInducingFeatures, ImageTranslateFeatures):
    """ HVGPs with the Translating transformation, and sharing the inducing points across groups """
    def __init__(self, z, step, num_translate, patch_shape, name='imagetranslation'):
        gpflow.features.InducingFeature.__init__(self, name=name)
        self.Z = Parameter(z, dtype=settings.float_type)
        self.patch_shape = patch_shape

        self.num_translate = num_translate
        self.step = step
        self.groups_per_direction = int(np.floor(num_translate / 2.) + 1)
        self.num_groups = self.groups_per_direction ** 2

        self._setup_DFT_matrix()
        self.Z.build()
        self.Z_orbit = self.orbit(self.Z.constrained_tensor)
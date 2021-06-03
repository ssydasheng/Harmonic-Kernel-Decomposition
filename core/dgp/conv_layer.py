"""
Adapted from https://github.com/kekeblom/DeepCGP
"""

import numpy as np
import tensorflow as tf
from gpflow import settings
from gpflow import transforms
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.params import Parameter
from gpflow.conditionals import Kuu, Kuf

from core.dgp.utils.kernels import MultiOutputConvKernel
from core.utils.conditional import base_conditional, blk_mvg_conditional
from core.utils.distributions import UnivariateGaussian
from core.utils.kl import compute_mvg_kl_divergence
from core.utils.distributions import LeadingDimRowIndMVG2, SumGaussian
from .dense_layer import Layer

float_type = settings.float_type
logger = settings.logger()


class Conv_Layer(Layer):
    def __init__(self, base_kern, mean_function, feature=None, view=None,
                 white=False, q_sqrt_init_scale=1., gp_count=1, pool=None, pool_size=2,
                 name='convlayer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_kern = base_kern
        self.view = view

        self.feature_maps_in = self.view.feature_maps
        self.gp_count = gp_count

        self.patch_count = self.view.patch_count
        self.patch_length = self.view.patch_length
        self.num_outputs = self.patch_count * gp_count

        self.conv_kernel = MultiOutputConvKernel(
            base_kern, self.view, patch_count=self.patch_count, name=name)

        self.white = white
        self.feature = feature
        self.num_inducing = len(feature)

        self.q_sqrt_init_scale = q_sqrt_init_scale
        self.mean_function = mean_function

        self.pool = pool
        self.pool_size = pool_size

        self._build_prior_cholesky()
        self._initialize_variational()

    def sample_from_conditional(self, X, full_cov=False):
        samples, dist = super().sample_from_conditional(X, full_cov)
        if self.pool is not None:
            N = tf.shape(samples)[0]
            height, width = self.view._out_image_size()
            NHWC_X = tf.reshape(samples, [N, height, width, self.gp_count])
            pool_func = dict(max=tf.layers.max_pooling2d, mean=tf.layers.average_pooling2d)[self.pool]
            NHWC_X = pool_func(NHWC_X, self.pool_size, self.pool_size, "SAME")
            samples = tf.reshape(NHWC_X, [N, np.prod(NHWC_X.shape.as_list()[1:])])
        return samples, dist

    @params_as_tensors
    def _build_prior_cholesky(self):
        self.Ku = Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)
        self.Lu = tf.linalg.cholesky(self.Ku)

class Conv_SVGP_Layer(Conv_Layer):
    pass

class Conv_KFGD_Layer(Conv_Layer):

    def _initialize_variational(self):
        q_mu = np.zeros((self.num_inducing, self.gp_count))
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)

        if not self.white:
            MM_Ku = Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)
            MM_Lu = tf.linalg.cholesky(MM_Ku)
            MM_Lu = self.enquire_session().run(MM_Lu)
        else:
            MM_Lu = np.eye(self.num_inducing, dtype=float_type)

        transform = transforms.LowerTriangular(self.num_inducing, squeeze=True)
        self.cov_u_sqrt = Parameter(MM_Lu, transform=transform, dtype=settings.float_type)

        transform = transforms.LowerTriangular(self.gp_count, squeeze=True)
        eye = np.eye(self.gp_count, dtype=settings.float_type)
        self.cov_v_sqrt = Parameter(self.q_sqrt_init_scale * eye, transform=transform,
                                    dtype=settings.float_type)

    def conditional_SND(self, X, full_cov=False):
        if full_cov is True:
            raise NotImplementedError
        else:
            # [P, N, R], [P, N], [P, N, R], [P, N]
            para_mean, para_u_cov, orth_mean, orth_u_cov = self.conditional_ND(X)
            para_dist = LeadingDimRowIndMVG2(
                para_mean[:, None],
                var_u=para_u_cov[:, None],
                cov_v_sqrt=self.cov_v_sqrt[None, ...])
            orth_dist = LeadingDimRowIndMVG2(
                orth_mean[:, None],
                var_u=orth_u_cov[:, None],
                cov_v_sqrt=tf.eye(self.gp_count, dtype=settings.float_type)[None, ...])

            return SumGaussian([para_dist, orth_dist])

    def conditional_ND(self, ND_X, full_cov=False):

        N = tf.shape(ND_X)[0]
        NHWC_X = tf.reshape(ND_X, [N, self.view.input_size[0], self.view.input_size[1], self.feature_maps_in])

        # [P, M, N]
        PMN_Kuf = Kuf(self.feature, self.conv_kernel, NHWC_X)
        P, R = PMN_Kuf.get_shape().as_list()[0], self.gp_count

        if full_cov:
            raise NotImplementedError
        else:
            Knn = self.conv_kernel.Kdiag(NHWC_X)

        # [P, N, R], [P, N], [P, N]
        para_mean, para_u_cov, orth_u_cov = blk_mvg_conditional(
            PMN_Kuf, self.Ku, Knn,
            self.q_mu,
            full_cov=full_cov,
            cov_us_sqrt=self.cov_u_sqrt,
            white=self.white,
            Lms=self.Lu)

        if not isinstance(self.mean_function, Zero):
            mean_func = self.mean_function(NHWC_X) # [N, P*R]
            orth_mean = tf.transpose(tf.reshape(mean_func, [N, P, R]), [1, 0, 2])
        else:
            orth_mean = tf.zeros([P, N, R], dtype=settings.float_type)

        # [P, N, R], [P, N], [P, N, R], [P, N]
        return para_mean, para_u_cov, orth_mean, orth_u_cov

    def KL(self):
        if not self.white:
            Lu = self.Lu
        else:
            Lu = tf.eye(self.num_inducing, dtype=float_type)
        kl = compute_mvg_kl_divergence(
            (self.q_mu, self.cov_u_sqrt, self.cov_v_sqrt),
            (tf.zeros_like(self.q_mu), Lu, tf.eye(self.gp_count, dtype=settings.float_type)),
            jitter=1e-8, sqrt_u1=True, sqrt_v1=True, sqrt_u2=True, sqrt_v2=True,
            lower_u1=True, lower_v1=True)
        return kl


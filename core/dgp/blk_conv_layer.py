"""
Adapted from https://github.com/kekeblom/DeepCGP
"""

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import settings, params_as_tensors, ParamList
from gpflow.params import Parameter
from gpflow.mean_functions import Zero
from gpflow import transforms
from gpflow.conditionals import Kuu, Kuf

from .conv_layer import Conv_Layer
from core.utils.distributions import SumGaussian, LeadingDimRowIndMVG2
from core.utils.kl import compute_mvg_kl_divergence
from core.utils.conditional import blk_mvg_conditional

float_type = settings.float_type
logger = settings.logger()


class BlockConvKFGD_Layer(Conv_Layer):
    def __init__(self, base_kern, mean_function, feature, view=None,
                 white=False, q_sqrt_init_scale=1., gp_count=1, pool=None, pool_size=2,
                 **kwargs):
        super().__init__(
            base_kern=base_kern,
            mean_function=mean_function,
            feature=feature,
            view=view,
            white=white,
            q_sqrt_init_scale=q_sqrt_init_scale,
            gp_count=gp_count,
            pool=pool,
            pool_size=pool_size,
            **kwargs)

    @params_as_tensors
    def _build_prior_cholesky(self):
        self.Kus = Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)
        self.Lus = tf.linalg.cholesky(self.Kus)

        self.J = self.Kus.get_shape().as_list()[0]
        self.M = self.Kus.get_shape().as_list()[1]

    def _initialize_variational(self):
        qs_mu = np.zeros((self.J, self.M, self.gp_count))
        self.qs_mu = Parameter(qs_mu, dtype=settings.float_type)

        self.unpacked_cov_us_sqrt = ParamList([])
        self.unpacked_cov_vs_sqrt = ParamList([])
        transform_u = transforms.LowerTriangular(self.M, squeeze=True)
        transform_v = transforms.LowerTriangular(self.gp_count, squeeze=True)
        eye_u = np.eye(self.M, dtype=settings.float_type)
        eye_v = np.eye(self.gp_count, dtype=settings.float_type) * self.q_sqrt_init_scale
        if not self.white:
            np_Lus = self.enquire_session().run(self.Lus)
        for j in range(self.J):
            if self.white:
                sqrt_u = eye_u
            else:
                sqrt_u = np_Lus[j]

            sqrt_u = Parameter(sqrt_u, transform=transform_u, dtype=settings.float_type)
            sqrt_v = Parameter(eye_v, transform=transform_v, dtype=settings.float_type)
            self.unpacked_cov_us_sqrt.append(sqrt_u)
            self.unpacked_cov_vs_sqrt.append(sqrt_v)

    @property
    @params_as_tensors
    def cov_us_sqrt(self):
        return tf.stack([a.constrained_tensor for a in self.unpacked_cov_us_sqrt.params])

    @property
    @params_as_tensors
    def cov_vs_sqrt(self):
        return tf.stack([a.constrained_tensor for a in self.unpacked_cov_vs_sqrt.params])

    def conditional_SND(self, X, full_cov=False):
        if full_cov is True:
            raise NotImplementedError
        else:
            # [P, J, N, R], [P, J, N], [P, N, R], [P, N]
            para_mean, para_u_cov, orth_mean, orth_u_cov = self.conditional_ND(X)
            para_dist = LeadingDimRowIndMVG2(
                para_mean,
                var_u=para_u_cov,
                cov_v_sqrt=self.cov_vs_sqrt)
            orth_dist = LeadingDimRowIndMVG2(
                orth_mean[:, None],
                var_u=orth_u_cov[:, None],
                cov_v_sqrt=tf.eye(self.gp_count, dtype=settings.float_type)[None, ...]
            )
            return SumGaussian([para_dist, orth_dist])

    @params_as_tensors
    def compute_PJMN_Kuf(self, NHWC_X):
        return Kuf(self.feature, self.conv_kernel, NHWC_X)

    def conditional_ND(self, ND_X, full_cov=False):
        """
        Returns the mean and the variance of q(f|m, S) = N(f| Am, K_nn + A(S - K_mm)A^T)
        where A = K_nm @ K_mm^{-1}

        dimension O: num_outputs (== patch_count * gp_count)

        :param ON_X: The input X of shape O x N
        :param full_cov: if true, var is in (N, N, D_out) instead of (N, D_out) i.e. we
        also compute entries outside the diagonal.
        """
        N = tf.shape(ND_X)[0]
        NHWC_X = tf.reshape(ND_X, [N, self.view.input_size[0], self.view.input_size[1], self.feature_maps_in])

        # [P, J, M, N]
        PJMN_Kuf = self.compute_PJMN_Kuf(NHWC_X)
        P, R = PJMN_Kuf.get_shape().as_list()[0], self.gp_count

        if full_cov:
            raise NotImplementedError
        else:
            Knn = self.conv_kernel.Kdiag(NHWC_X)

        # [P, J, N, R], [P, J, N], [P, N]
        para_mean, para_u_cov, orth_u_cov = blk_mvg_conditional(
            PJMN_Kuf, self.Kus, Knn,
            qs_mu=self.qs_mu,
            cov_us_sqrt=self.cov_us_sqrt,
            full_cov=full_cov,
            white=self.white,
            Lms=self.Lus)

        if not isinstance(self.mean_function, Zero):
            mean_func = tf.reshape(self.mean_function(NHWC_X), [N, P, R]) # [N, P*R]
            orth_mean = tf.transpose(mean_func, [1, 0, 2])
        else:
            orth_mean = tf.zeros([P, N, R], dtype=settings.float_type)

        # [P, J, N, R], [P, J, N], [P, N, R], [P, N]
        return para_mean, para_u_cov, orth_mean, orth_u_cov

    def KL(self):
        eye_v = tf.tile(tf.eye(self.gp_count, dtype=settings.float_type)[None], [self.J, 1, 1])
        eye_u = tf.tile(tf.eye(self.M, dtype=settings.float_type)[None], [self.J, 1, 1])
        if not self.white: Lu = self.Lus
        else: Lu = eye_u
        return compute_mvg_kl_divergence(
            (self.qs_mu, self.cov_us_sqrt, self.cov_vs_sqrt),
            (tf.zeros_like(self.qs_mu), Lu, eye_v),
            jitter=1e-8, sqrt_u1=True, sqrt_v1=True, sqrt_u2=True, sqrt_v2=True,
            lower_u1=True, lower_v1=True)


class SOLVEGP_CONV_KFGD_Layer(BlockConvKFGD_Layer):
    @params_as_tensors
    def _build_prior_cholesky(self):
        # init Kernel matrix
        K = Kuu(self.feature, self.conv_kernel, jitter=settings.jitter)  # [2M, 2M]
        self.M = int(self.num_inducing / 2.)
        assert self.num_inducing == 2 * self.M
        self.J = 2

        Ku, Kuv, Kv = K[:self.M, :self.M], K[:self.M, self.M:], K[self.M:, self.M:]
        self.Lu_perp = tf.linalg.cholesky(Ku)
        self.Lu_inv_Kuv = tf.linalg.triangular_solve(self.Lu_perp, Kuv, lower=True)
        Cvv = Kv - tf.matmul(self.Lu_inv_Kuv, self.Lu_inv_Kuv, transpose_a=True)
        Lv = tf.linalg.cholesky(Cvv)

        self.Kus = tf.stack([Ku, Cvv])
        self.Lus = tf.stack([self.Lu_perp, Lv])

    @params_as_tensors
    def compute_PJMN_Kuf(self, NHWC_X):
        K = Kuf(self.feature, self.conv_kernel, NHWC_X)

        Kux, Kvx = K[:, :self.M], K[:, self.M:] # [P, M, N]
        Lu_perp_inv = tf.matrix_triangular_solve(
            self.Lu_perp, tf.eye(tf.shape(self.Lu_perp)[-1], dtype=self.Lu_perp.dtype))
        Lu_inv_Kux = tf.matmul(Lu_perp_inv, Kux) # [P, M, N]
        Cvx = Kvx - tf.matmul(self.Lu_inv_Kuv, Lu_inv_Kux, transpose_a=True) # [P, M, N]
        return tf.concat([Kux[:, None], Cvx[:, None]], axis=1)
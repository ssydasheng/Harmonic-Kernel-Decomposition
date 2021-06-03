# Copyright 2017 Hugh Salimbeni, 2020 Shengyang Sun
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from gpflow import params_as_tensors, name_scope
from gpflow import settings
from gpflow import transforms
from gpflow.features import Kuf as Kern_u_f
from gpflow.features import Kuu as Kern_u_u
from gpflow.params import Parameter, Parameterized

from core.utils.conditional import blk_mvg_conditional
from core.utils.distributions import SumGaussian
from core.utils.distributions import UnivariateGaussian, RowIndMVG
from core.utils.kl import compute_mvg_kl_divergence, compute_mvg_kl_divergence_whiten
from core.utils.utils import InducingPoints


class Layer(Parameterized):
    def __init__(self, name='Layer', **kwargs):
        """
        A base class for GP layers. Basic functionality for multisample conditional, and input propagation
        :param kwargs:
        """
        Parameterized.__init__(self, name=name)

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        raise NotImplementedError

    def conditional_SND(self, X, full_cov=False):
        raise NotImplementedError

    @params_as_tensors
    def sample_from_conditional(self, X, full_cov=False):
        """
        Calculates self.conditional and also draws a sample, adding input propagation if necessary

        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points

        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :return: mean (S,N,D), var (S,N,N,D or S,N,D), samples (S,N,D), invKuu_Kuf (S, N, M)
        """
        with tf.name_scope('sample_%s' % self.name):
            dist = self.conditional_SND(X, full_cov=full_cov)
            samples = dist.sample()
            return samples, dist

    @params_as_tensors
    def build_cholesky(self):
        if self.needs_build_cholesky:
            self.Ku = Kern_u_u(self.feature, self.kern, jitter=settings.jitter)
            self.Lu = tf.cholesky(self.Ku)

            self.Ku_tiled = tf.tile(self.Ku[None, :, :], [self.num_outputs, 1, 1])
            self.Lu_tiled = tf.tile(self.Lu[None, :, :], [self.num_outputs, 1, 1])
            self.needs_build_cholesky = False


class SVGP_Layer(Layer):
    def __init__(self, kern, num_outputs, mean_function, Z=None, feature=None,
                 white=False, q_sqrt_init_scale=1.,
                 **kwargs):
        """
        A sparse variational GP layer in whitened representation. This layer holds the kernel,
        variational parameters, inducing points and mean function.

        The underlying model at inputs X is
        f = Lv + mean_function(X), where v \sim N(0, I) and LL^T = kern.K(X)

        The variational distribution over the inducing points is
        q(v) = N(q_mu, q_sqrt q_sqrt^T)

        The layer holds D_out independent GPs with the same kernel and inducing points.

        :param kern: The kernel for the layer (input_dim = D_in)
        :param Z: Inducing points (M, D_in)
        :param num_outputs: The number of GP outputs (q_mu is shape (M, num_outputs))
        :param mean_function: The mean function
        :param q_sqrt_init_scale: float. Init value for q_sqrt.
        :return:
        """
        Layer.__init__(self, **kwargs)
        assert feature is not None

        self.num_inducing = len(feature)
        self.feature = feature

        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)

        q_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [num_outputs, 1, 1])
        transform = transforms.LowerTriangular(self.num_inducing, num_matrices=num_outputs)
        self.q_sqrt = Parameter(q_sqrt * q_sqrt_init_scale, transform=transform, dtype=settings.float_type)

        self.kern = kern
        self.mean_function = mean_function

        self.num_outputs = num_outputs
        self.white = white

        if not self.white:  # initialize to prior
            if Z is not None:
                Ku = kern.compute_K_symm(Z) + np.eye(Z.shape[0]) * settings.jitter
            else:
                Ku = Kern_u_u(self.feature, kern, jitter=settings.jitter)
                Ku = self.enquire_session().run(Ku)
            Lu = np.linalg.cholesky(Ku)
            self.q_sqrt = np.tile(Lu[None, :, :], [num_outputs, 1, 1]) * q_sqrt_init_scale

        self.needs_build_cholesky = True

    def conditional_SND(self, X, full_cov=False):
        """
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S

        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)

        if full_cov is False
            mean and var are both (S,N,D_out)

        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out), invKuu_Kuf (S, N, M)
        """
        if full_cov is True:
            raise NotImplementedError
        else:
            mean, var = self.conditional_ND(X)
            return UnivariateGaussian(mean, var)

    def conditional_ND(self, X, full_cov=False):
        self.build_cholesky()

        # mmean, vvar = conditional(X, self.feature.Z, self.kern,
        #             self.q_mu, q_sqrt=self.q_sqrt,
        #             full_cov=full_cov, white=self.white)
        Kuf = Kern_u_f(self.feature, self.kern, X)

        A = tf.matrix_triangular_solve(self.Lu, Kuf, lower=True)
        invKuu_Kuf = tf.matrix_triangular_solve(tf.transpose(self.Lu), A, lower=False)
        if not self.white:
            A = invKuu_Kuf

        mean = tf.matmul(A, self.q_mu, transpose_a=True)

        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1])
        I = tf.eye(self.num_inducing, dtype=settings.float_type)[None, :, :]

        if self.white:
            SK = -I
        else:
            SK = -self.Ku_tiled

        if self.q_sqrt is not None:
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True)


        B = tf.matmul(SK, A_tiled)

        if full_cov:
            # (num_latent, num_X, num_X)
            delta_cov = tf.matmul(A_tiled, B, transpose_a=True)
            Kff = self.kern.K(X)
        else:
            # (num_latent, num_X)
            delta_cov = tf.reduce_sum(A_tiled * B, 1)
            Kff = self.kern.Kdiag(X)

        # either (1, num_X) + (num_latent, num_X) or (1, num_X, num_X) + (num_latent, num_X, num_X)
        var = tf.expand_dims(Kff, 0) + delta_cov
        var = tf.transpose(var)

        return mean + self.mean_function(X), var

    @params_as_tensors
    def KL(self):
        KL = -0.5 * self.num_outputs * self.num_inducing
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.q_sqrt) ** 2))

        if not self.white:
            KL += tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Lu))) * self.num_outputs
            KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.Lu_tiled, self.q_sqrt, lower=True)))
            Kinv_m = tf.cholesky_solve(self.Lu, self.q_mu)
            KL += 0.5 * tf.reduce_sum(self.q_mu * Kinv_m)
        else:
            KL += 0.5 * tf.reduce_sum(tf.square(self.q_sqrt))
            KL += 0.5 * tf.reduce_sum(self.q_mu**2)

        return KL


class SVGP_KFGD_Layer(Layer):
    """
    SVGP Layer using Kronecker-Factored Covariance for the Variational Distribution
    """
    def __init__(self, kern, num_outputs, mean_function, Z=None, feature=None,
                 white=False, q_sqrt_init_scale=1., **kwargs):
        Layer.__init__(self, **kwargs)
        assert feature is not None

        self.num_inducing = len(feature)
        self.feature = feature

        self.kern = kern
        self.mean_function = mean_function
        self.num_outputs = num_outputs

        self.white = white
        self.needs_build_cholesky = True
        self.q_sqrt_init_scale = q_sqrt_init_scale

        self._initialize_variational(Z)

    def _initialize_variational(self, Z):
        q_mu = np.zeros((self.num_inducing, self.num_outputs))
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)

        transform = transforms.LowerTriangular(self.num_inducing, squeeze=True)
        eye = np.eye(self.num_inducing, dtype=settings.float_type)
        self.cov_u_sqrt = Parameter(eye, transform=transform, dtype=settings.float_type)
        transform = transforms.LowerTriangular(self.num_outputs, squeeze=True)
        eye = np.eye(self.num_outputs, dtype=settings.float_type)
        self.cov_v_sqrt = Parameter(self.q_sqrt_init_scale * eye, transform=transform,
                                    dtype=settings.float_type)

        if not self.white:
            if Z is not None:
                Ku = self.kern.compute_K_symm(Z) + np.eye(Z.shape[0]) * settings.jitter
            else:
                Ku = Kern_u_u(self.feature, self.kern, jitter=settings.jitter)
                Ku = self.enquire_session().run(Ku)
            Lu = np.linalg.cholesky(Ku)
            self.cov_u_sqrt = Lu

    def conditional_ND(self, X, full_cov=False):
        # X: [N, D]
        self.build_cholesky()

        para_mean, para_u_cov, orth_u_cov = blk_mvg_conditional(
            Kern_u_f(self.feature, self.kern, X),
            self.Ku,
            self.kern.K(X) if full_cov else self.kern.Kdiag(X),
            qs_mu=self.q_mu,
            cov_us_sqrt=self.cov_u_sqrt,
            full_cov=full_cov,
            white=self.white,
            Lms=self.Lu
        )
        orth_mean = self.mean_function(X)  # [N, D]
        return para_mean, para_u_cov, orth_mean, orth_u_cov

    def conditional_SND(self, X, full_cov=False):
        if full_cov is True:
            raise NotImplementedError
        else:
            para_mean, para_u_cov, orth_mean, orth_u_cov = self.conditional_ND(X)
            para_dist = RowIndMVG(
                para_mean[None, ...],
                var_u=para_u_cov[None, ...],
                cov_v_sqrt=self.cov_v_sqrt[None, ...])
            orth_dist = RowIndMVG(
                orth_mean[None, ...],
                var_u=orth_u_cov[None, ...],
                cov_v_sqrt=tf.eye(self.num_outputs, dtype=settings.float_type)[None, ...])
            return SumGaussian([para_dist, orth_dist])

    @params_as_tensors
    def KL(self):
        if self.white:
            kl = compute_mvg_kl_divergence_whiten(
                (self.q_mu, self.cov_u_sqrt, self.cov_v_sqrt),
                jitter=1e-8, sqrt_u1=True, sqrt_v1=True)
        else:
            kl = compute_mvg_kl_divergence(
                (self.q_mu, self.cov_u_sqrt, self.cov_v_sqrt),
                (tf.zeros_like(self.q_mu), self.Lu, tf.eye(self.num_outputs, dtype=settings.float_type)),
                jitter=1e-8, sqrt_u1=True, sqrt_v1=True, sqrt_u2=True, sqrt_v2=True,
                lower_u1=True, lower_v1=True)
        return kl


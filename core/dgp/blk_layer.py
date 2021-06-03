# Copyright 2020 Shengyang Sun
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
from gpflow.params import Parameter, ParamList

from core.utils.conditional import blk_mvg_conditional
from core.utils.distributions import RowIndMVG
from core.utils.distributions import SumGaussian
from core.utils.kl import compute_mvg_kl_divergence, compute_mvg_kl_divergence_whiten
from core.utils.utils import BatchLowerTriangular
from .dense_layer import Layer


class BlockSVGP_KFGD_Layer(Layer):
    def __init__(self, kern, num_outputs, mean_function, feature,
                 white=False, q_sqrt_init_scale=1., **kwargs):
        Layer.__init__(self, **kwargs)

        self.num_inducing = len(feature)
        self.feature = feature

        self.kern = kern
        self.mean_function = mean_function
        self.num_outputs = num_outputs
        self.white = white
        self.q_sqrt_init_scale = q_sqrt_init_scale

        self._build_prior_cholesky()
        self._initialize_variational()

    @params_as_tensors
    def _build_prior_cholesky(self):
        # init Kernel matrix
        self.Kus = Kern_u_u(self.feature, self.kern, jitter=settings.jitter)  # [J, M, M]
        self.J = self.Kus.get_shape().as_list()[0]
        self.M = self.Kus.get_shape().as_list()[1]
        self.Lus = tf.cholesky(self.Kus)  # [J, M, M]

    @name_scope('initialize')
    def _initialize_variational(self):
        mu = np.zeros((self.J, self.M, self.num_outputs))
        self.qs_mu = Parameter(mu, dtype=settings.float_type)  # J x M x P

        transform_u = BatchLowerTriangular(self.M)
        transform_v = BatchLowerTriangular(self.num_outputs)
        if not self.white:
            np_Lus = self.enquire_session().run(self.Lus)
        else:
            np_Lus = np.tile(np.eye(self.M, dtype=settings.float_type)[None], [self.J, 1, 1])
        self.cov_us_sqrt = Parameter(np_Lus, transform=transform_u, dtype=settings.float_type)
        eye_v = np.eye(self.num_outputs, dtype=settings.float_type) * self.q_sqrt_init_scale
        eye_v = np.tile(eye_v[None], [self.J, 1, 1])
        self.cov_vs_sqrt = Parameter(eye_v, transform=transform_v, dtype=settings.float_type)

    @params_as_tensors
    def compute_Kzx(self, X):
        return Kern_u_f(self.feature, self.kern, X)

    @name_scope('conditional_ND')
    @params_as_tensors
    def conditional_ND(self, X, full_cov=False):
        Kzx = self.compute_Kzx(X)
        para_mean, para_u_cov, orth_u_cov = blk_mvg_conditional(
            Kzx,
            self.Kus,
            self.kern.K(X) if full_cov else self.kern.Kdiag(X),
            qs_mu=self.qs_mu,
            cov_us_sqrt=self.cov_us_sqrt,
            full_cov=full_cov,
            white=self.white,
            Lms=self.Lus
        )
        orth_mean = self.mean_function(X) # [N, R]
        # [J, N, R], [J, N], [N, R], [N]
        return para_mean, para_u_cov, orth_mean, orth_u_cov

    @name_scope('conditional_SND')
    def conditional_SND(self, X, full_cov=False):
        if full_cov is True:
            raise NotImplementedError
        else:
            para_mean, para_u_cov, orth_mean, orth_u_cov = self.conditional_ND(X)
            para_dist = RowIndMVG(
                para_mean,
                var_u=para_u_cov,
                cov_v_sqrt=self.cov_vs_sqrt)
            orth_dist = RowIndMVG(
                orth_mean[None,...],
                var_u=orth_u_cov[None,...],
                cov_v_sqrt=tf.eye(self.num_outputs, dtype=settings.float_type)[None, ...])
            return SumGaussian([para_dist, orth_dist])

    @name_scope('KL')
    @params_as_tensors
    def KL(self):
        if self.white:
            return compute_mvg_kl_divergence_whiten(
                (self.qs_mu, self.cov_us_sqrt, self.cov_vs_sqrt),
                jitter=1e-8, sqrt_u1=True, sqrt_v1=True
            )
        else:
            eye = tf.tile(tf.eye(self.num_outputs, dtype=settings.float_type)[None], [self.J, 1, 1])
            return compute_mvg_kl_divergence(
                (self.qs_mu, self.cov_us_sqrt, self.cov_vs_sqrt),
                (tf.zeros_like(self.qs_mu), self.Lus, eye),
                jitter=1e-8, sqrt_u1=True, sqrt_v1=True, sqrt_u2=True, sqrt_v2=True,
                lower_u1=True, lower_v1=True)


class SOLVEGP_KFGD_Layer(BlockSVGP_KFGD_Layer):
    @params_as_tensors
    def _build_prior_cholesky(self):
        # init Kernel matrix
        K = Kern_u_u(self.feature, self.kern, jitter=settings.jitter)  # [2M, 2M]
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
    def compute_Kzx(self, X):
        K = Kern_u_f(self.feature, self.kern, X)
        Kux, Kvx = K[:self.M], K[self.M:]
        Lu_inv_Kux = tf.matrix_triangular_solve(self.Lu_perp, Kux)
        Cvx = Kvx - tf.matmul(self.Lu_inv_Kuv, Lu_inv_Kux, transpose_a=True)
        return tf.stack([Kux, Cvx])


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

import gpflow
from gpflow import params_as_tensors, name_scope
from gpflow import settings
from gpflow import transforms
from gpflow.features import Kuf as Kern_u_f
from gpflow.features import Kuu as Kern_u_u
from gpflow.params import Parameter, ParamList
from gpflow.mean_functions import Zero

from core.utils.conditional import blk_mvg_conditional
from core.utils.distributions import MGRowIndMVG
from core.utils.distributions import SumGaussian
from core.utils.utils import BatchLowerTriangular, none_sum
from ..dense_layer import Layer, SVGP_KFGD_Layer


class MultiGPU_KFGD_Layer(Layer):
    def __init__(self, kern, num_outputs, mean_function, feature,
                 white=False, q_sqrt_init_scale=1., **kwargs):
        Layer.__init__(self, **kwargs)

        self.kern = kern
        cls_and_Z = feature
        self.layers = ParamList([])
        num_gpu = len(cls_and_Z)
        for i in range(num_gpu):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("gpu_%d" % i):
                    feature_cls, Z = cls_and_Z[i]
                    feat = feature_cls(Z)
                    feat.build()
                    feat.initialize()
                    layer = SVGP_KFGD_Layer(
                        kern, num_outputs, Zero(), feature=feat, white=white,
                        q_sqrt_init_scale=q_sqrt_init_scale, name='layer_%d' % i)
                    self.layers.append(layer)
                    layer.build()

        self.cov_vs_sqrt = tf.stack([l.cov_v_sqrt.constrained_tensor for l in self.layers])
        self.mean_function = mean_function
        self.white = white
        self.num_gpu = num_gpu
        self.num_outputs = num_outputs

    @params_as_tensors
    def _conditional_ND_group(self, X, group_idx, full_cov=False):
        with tf.device("/gpu:%d" % group_idx):
            layer = self.layers[group_idx]
            Kzx = Kern_u_f(layer.feature, layer.kern, X)
            layer.build_cholesky()
            para_mean, para_u_cov, orth_u_cov = blk_mvg_conditional(
                Kzx,
                layer.Ku,
                None,
                qs_mu=layer.q_mu,
                cov_us_sqrt=layer.cov_u_sqrt,
                full_cov=full_cov,
                white=self.white,
                Lms=layer.Lu,
                return_nystrom_residual=False
            )
            return para_mean, para_u_cov, orth_u_cov

    @name_scope('conditional_ND')
    @params_as_tensors
    def conditional_ND(self, X, full_cov=False):
        self.Knn = self.kern.K(X) if full_cov else self.kern.Kdiag(X)
        self.all_para_mean, self.all_para_u_cov, self.all_orth_u_cov = [], [], []
        for group_idx in range(self.num_gpu):
            para_mean, para_u_cov, orth_u_cov = self._conditional_ND_group(X, group_idx, full_cov)
            self.all_para_mean.append(para_mean)
            self.all_para_u_cov.append(para_u_cov)
            self.all_orth_u_cov.append(orth_u_cov)

        self.orth_mean = self.mean_function(X) # [N, R]
        self.orth_u_cov = self.Knn - tf.add_n(self.all_orth_u_cov)
        para_mean = tf.stack(self.all_para_mean)
        para_u_cov = tf.stack(self.all_para_u_cov)
        return para_mean, para_u_cov, self.orth_mean, self.orth_u_cov

    @params_as_tensors
    def sample_from_conditional(self, X, full_cov=False):
        with tf.name_scope('sample_%s' % self.name):
            self.dist = self.conditional_SND(X, full_cov=full_cov)
            self.samples = self.dist.sample()
            return self.samples, self.dist

    @name_scope('conditional_SND')
    def conditional_SND(self, X, full_cov=False):
        if full_cov is True:
            raise NotImplementedError
        else:
            para_mean, para_u_cov, orth_mean, orth_u_cov = self.conditional_ND(X)
            para_dist = MGRowIndMVG(
                para_mean,
                var_u=para_u_cov,
                cov_v_sqrt=self.cov_vs_sqrt)
            orth_dist = MGRowIndMVG(
                orth_mean[None,...],
                var_u=orth_u_cov[None,...],
                cov_v_sqrt=tf.eye(self.num_outputs, dtype=settings.float_type)[None, ...])
            return SumGaussian([para_dist, orth_dist])

    @name_scope('backprop')
    @params_as_tensors
    def backprop(self, vars, layer_input, grad_output_samples, grad_output_dist, grad_KLs):
        all_bp_tensors = self.all_para_mean + self.all_para_u_cov + self.all_orth_u_cov \
                         + [self.cov_vs_sqrt, self.Knn, self.orth_mean]

        # using params_after/before_detach is for preventing from computing the gradients in previous layers
        if grad_output_dist is None:
            assert grad_output_samples is not None
            grad_output_dist = tf.gradients(self.samples, self.dist.params_after_detach, grad_ys=grad_output_samples)
        all_bp_grads = tf.gradients(self.dist.params_before_detach, all_bp_tensors, grad_ys=grad_output_dist)

        all_para_mean_grads = all_bp_grads[ : self.num_gpu]
        all_para_u_cov_grads = all_bp_grads[self.num_gpu : 2*self.num_gpu]
        all_orth_u_cov_grads = all_bp_grads[2*self.num_gpu : 3*self.num_gpu]
        all_other_grads = all_bp_grads[3*self.num_gpu : ]

        if layer_input is None:
            all_bp_vars = vars
        else:
            all_bp_vars = vars + [layer_input]
        grad_others = tf.gradients(all_bp_tensors[3*self.num_gpu:], all_bp_vars, grad_ys=all_other_grads)
        grad_devices = []
        for group_idx in range(self.num_gpu):
            with tf.device("/gpu:%d" % group_idx):
                grad = tf.gradients(
                    [self.all_para_mean[group_idx], self.all_para_u_cov[group_idx],
                     self.all_orth_u_cov[group_idx], self.KLs[group_idx]],
                    all_bp_vars,
                    grad_ys=[all_para_mean_grads[group_idx], all_para_u_cov_grads[group_idx],
                             all_orth_u_cov_grads[group_idx], grad_KLs[group_idx]]
                )
                grad_devices.append(grad)
        grad_vars = [none_sum(*a) for a in zip(*grad_devices, grad_others)]

        if layer_input is not None:
            grad_params, grad_input = grad_vars[:-1], grad_vars[-1]
        else:
            grad_params, grad_input = grad_vars, None
        return grad_params, grad_input

    @name_scope('KL')
    @params_as_tensors
    def KL(self):
        self.KLs = []
        for group_idx in range(self.num_gpu):
            with tf.device("/gpu:%d" % group_idx):
                self.KLs.append(self.layers[group_idx].KL())
        return tf.add_n(self.KLs)
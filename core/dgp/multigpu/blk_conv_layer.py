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

from core.dgp.dense_layer import Layer
from core.dgp.conv_layer import Conv_Layer, Conv_KFGD_Layer
from core.utils.distributions import SumGaussian, MGLeadingDimRowIndMVG2
from core.utils.conditional import blk_mvg_conditional
from core.dgp.multigpu.blk_layer import MultiGPU_KFGD_Layer
from core.dgp.utils.kernels import MultiOutputConvKernel

float_type = settings.float_type
logger = settings.logger()


class MultiGPU_ConvKFGD_Layer(Conv_Layer, MultiGPU_KFGD_Layer):
    def __init__(self, base_kern, mean_function, feature, view=None,
                 white=False, q_sqrt_init_scale=1., gp_count=1, pool=None, pool_size=2,
                 **kwargs):
        Layer.__init__(self, **kwargs)

        self.view = view
        cls_and_Z = feature
        self.layers = ParamList([])
        self.features = ParamList([])
        num_gpu = len(cls_and_Z)
        for i in range(num_gpu):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("gpu_%d" % i):
                    feature_cls, Z = cls_and_Z[i]
                    feat = feature_cls(Z)
                    feat.build()
                    feat.initialize()
                    self.features.append(feat)
                    self.layers.append(Conv_KFGD_Layer(
                        base_kern, Zero(), feature=feat, view=view,
                        white=white, q_sqrt_init_scale=q_sqrt_init_scale,
                        gp_count=gp_count, pool=pool, pool_size=pool_size, name='layer_%d' % i))
                    self.layers[-1].build()

        self.mean_function = mean_function
        self.white = white
        self.num_gpu = num_gpu
        self.feature_maps_in = self.view.feature_maps
        self.gp_count = gp_count
        self.patch_count = self.view.patch_count
        self.patch_length = self.view.patch_length
        self.num_outputs = self.patch_count * gp_count
        self.conv_kernel = MultiOutputConvKernel(
            base_kern, self.view, patch_count=self.patch_count, name=self.name)
        self.cov_vs_sqrt = tf.stack([l.cov_v_sqrt.constrained_tensor for l in self.layers])
        self.pool = pool
        self.pool_size = pool_size

    def sample_from_conditional(self, X, full_cov=False):
        self.samples, self.dist = super().sample_from_conditional(X, full_cov)
        return self.samples, self.dist

    def conditional_SND(self, X, full_cov=False):
        if full_cov is True:
            raise NotImplementedError
        else:
            # [P, J, N, R], [P, J, N], [P, N, R], [P, N]
            para_mean, para_u_cov, orth_mean, orth_u_cov = self.conditional_ND(X)
            para_dist = MGLeadingDimRowIndMVG2(
                para_mean,
                var_u=para_u_cov,
                cov_v_sqrt=self.cov_vs_sqrt)
            orth_dist = MGLeadingDimRowIndMVG2(
                orth_mean[:, None],
                var_u=orth_u_cov[:, None],
                cov_v_sqrt=tf.eye(self.gp_count, dtype=settings.float_type)[None, ...]
            )
            return SumGaussian([para_dist, orth_dist])

    @params_as_tensors
    def _conditional_ND_group(self, NHWC_X, group_idx, full_cov=False):
        with tf.device("/gpu:%d" % group_idx):
            layer = self.layers[group_idx]
            N = tf.shape(NHWC_X)[0]
            # [P, M, N]
            PMN_Kuf = Kuf(layer.feature, layer.conv_kernel, NHWC_X)

            # [P, N, R], [P, N], [P, N]
            para_mean, para_u_cov, orth_u_cov = blk_mvg_conditional(
                PMN_Kuf, layer.Ku, None,
                layer.q_mu,
                full_cov=full_cov,
                cov_us_sqrt=layer.cov_u_sqrt,
                white=layer.white,
                Lms=layer.Lu,
                return_nystrom_residual=False)
            # [P, N, R], [P, N], [P, N]
            return para_mean, para_u_cov, orth_u_cov

    @params_as_tensors
    def conditional_ND(self, ND_X, full_cov=False):
        N = tf.shape(ND_X)[0]
        NHWC_X = tf.reshape(ND_X, [N, self.view.input_size[0], self.view.input_size[1], self.feature_maps_in])

        if full_cov:
            raise NotImplementedError
        else:
            self.Knn = self.conv_kernel.Kdiag(NHWC_X)
        self.all_para_mean, self.all_para_u_cov, self.all_orth_u_cov = [], [], []
        for group_idx in range(self.num_gpu):
            para_mean, para_u_cov, orth_u_cov = self._conditional_ND_group(NHWC_X, group_idx, full_cov)
            self.all_para_mean.append(para_mean)
            self.all_para_u_cov.append(para_u_cov)
            self.all_orth_u_cov.append(orth_u_cov)

        P, R = self.all_para_mean[0].get_shape().as_list()[0], self.gp_count

        if not isinstance(self.mean_function, Zero):
            mean_func = tf.reshape(self.mean_function(NHWC_X), [N, P, R]) # [N, P*R]
            self.orth_mean = tf.transpose(mean_func, [1, 0, 2])
        else:
            self.orth_mean = tf.zeros([P, N, R], dtype=settings.float_type)

        self.orth_u_cov = self.Knn - tf.add_n(self.all_orth_u_cov)
        para_mean = tf.concat([a[:, None] for a in self.all_para_mean], axis=1)
        para_u_cov = tf.concat([a[:, None] for a in self.all_para_u_cov], axis=1)
        return para_mean, para_u_cov, self.orth_mean, self.orth_u_cov

    @params_as_tensors
    def KL(self):
        self.KLs = []
        for group_idx in range(self.num_gpu):
            with tf.device("/gpu:%d" % group_idx):
                self.KLs.append(self.layers[group_idx].KL())
        return tf.add_n(self.KLs)

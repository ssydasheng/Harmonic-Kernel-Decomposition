import tensorflow as tf
import gpflow
from gpflow import settings
from gpflow.conditionals import Kuf
from gpflow.models import GPModel
from gpflow.params import ParamList
from gpflow.conditionals import _expand_independent_outputs

from core.utils.conditional import blk_conditional
from core.svgp.svgp import SVGP


class SVGPWrapper(SVGP):
    def _build_likelihood(self):
        return tf.constant(0., dtype=settings.float_type)


class MultiGPU(GPModel):

    def __init__(self, X, Y, kern, likelihood, feat_fn, feat_Zs, num_gpu,
                 mean_function=None,
                 num_latent=None,
                 whiten=True,
                 num_data=None,
                 XY_tensor=False,
                 **kwargs):
        assert XY_tensor
        num_latent = num_latent or Y.get_shape().as_list()[1]
        gpflow.models.GPModel.__init__(
            self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)

        self.num_gps = len(feat_Zs)
        self.gp_to_gpu = [i % num_gpu for i in range(len(feat_Zs))]

        self.gps = ParamList([])
        for i in range(self.num_gps):
            with tf.device("/gpu:%d" % self.gp_to_gpu[i]):
                with tf.name_scope("tower_%d" % self.gp_to_gpu[i]):
                    feat = feat_fn(feat_Zs[i], i, 'feat_%d' % i)
                    feat.build()
                    self.gps.append(SVGPWrapper(
                        X, Y, kern, likelihood, feat,
                        whiten=whiten, num_data=num_data, num_latent=num_latent, name='gp_%d'%i))
                    self.gps[-1].build()

        self.whiten=whiten
        self.num_data=num_data
        self.num_gpu = num_gpu

    def _build_likelihood(self):
        return tf.constant(0., dtype=settings.float_type)

    @gpflow.params_as_tensors
    def _predict_group(self, Xnew, group_idx, full_cov=False):
        with tf.device("/gpu:%d" % self.gp_to_gpu[group_idx]):
            gp = self.gps[group_idx]
            with gpflow.params_as_tensors_for(gp, self.kern):
                Kmsn = Kuf(gp.feature, self.kern, Xnew)  # J x M x N
                fmean, fvar0, fvar = blk_conditional(
                    None,
                    Kmsn, None, gp.Lu, gp.q_mu, gp.q_sqrt,
                    full_cov=full_cov, white=self.whiten, return_nystrom_residual=False)
                return fmean, fvar0, fvar

    @gpflow.params_as_tensors
    def build_objective_and_grads(self):
        ### forward for each group
        Knn = self.kern.Kdiag(self.X) #TODO: BP HERE
        all_fmean, all_fvar0, all_fvar = [], [], []
        for group_idx in range(self.num_gps):
            fmean, fvar0, fvar = self._predict_group(self.X, group_idx, full_cov=False)
            all_fmean.append(fmean)
            all_fvar0.append(fvar0)
            all_fvar.append(fvar)

        ### combine all groups
        fmean0 = self.mean_function(self.X)
        fmean = tf.add_n(all_fmean) + fmean0
        fvar0 = Knn[..., None] - tf.add_n(all_fvar0)
        fvar = tf.add_n(all_fvar) + fvar0

        ### compute loss
        KLs = []
        for group_idx in range(self.num_gps):
            with tf.device("/gpu:%d" % self.gp_to_gpu[group_idx]):
                KLs.append(self.gps[group_idx].build_prior_KL())
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(
            tf.shape(self.X)[0], settings.float_type)
        elbo = tf.reduce_sum(var_exp) * scale - tf.add_n(KLs)
        loss = - elbo / self.num_data

        ### compute grad
        other_vars = tf.trainable_variables(self.likelihood.name) + tf.trainable_variables(self.mean_function.name)
        grad_KLs = tf.gradients(loss, KLs)
        grad_others = tf.gradients(loss, other_vars)
        grad_Knn = tf.gradients(loss, Knn)
        grad_all = tf.gradients(loss, all_fmean + all_fvar0 + all_fvar)
        grad_all_fmean = grad_all[:self.num_gps]
        grad_all_fvar0 = grad_all[self.num_gps: 2*self.num_gps]
        grad_all_fvar = grad_all[2*self.num_gps:]


        ### compute grad for each device
        grad_devices = []
        group_vars = list(set(tf.trainable_variables()) - set(other_vars))
        grad_Knn_group_vars = tf.gradients(Knn, group_vars, grad_ys=grad_Knn)
        for group_idx in range(self.num_gps):
            with tf.device("/gpu:%d" % self.gp_to_gpu[group_idx]):
                merged_tensor = tf.concat([all_fmean[group_idx],
                                           all_fvar0[group_idx],
                                           all_fvar[group_idx],
                                           tf.ones([1, 1], dtype=settings.float_type) * KLs[group_idx]], 0)
                merged_grad = tf.concat([grad_all_fmean[group_idx],
                                         grad_all_fvar0[group_idx],
                                         grad_all_fvar[group_idx],
                                         tf.ones([1, 1], dtype=settings.float_type) * grad_KLs[group_idx]], 0)
                grad = tf.gradients(merged_tensor, group_vars, grad_ys=merged_grad)
                grad_devices.append(grad)
        grad_group_vars = [none_sum(*a) for a in zip(*grad_devices, grad_Knn_group_vars)]
        grads_and_vars = [(g, v) for g, v in zip(grad_group_vars, group_vars)]\
                         + [(g, v) for g, v in zip(grad_others, other_vars)]
        return loss, grads_and_vars

    @gpflow.params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        ### forward for each group
        Knn = self.kern.Kdiag(Xnew)
        all_fmean, all_fvar0, all_fvar = [], [], []
        for group_idx in range(self.num_gps):
            fmean, fvar0, fvar = self._predict_group(Xnew, group_idx, full_cov=full_cov)
            all_fmean.append(fmean)
            all_fvar0.append(fvar0)
            all_fvar.append(fvar)

        ### combine all groups
        fmean0 = self.mean_function(Xnew)
        fmean = tf.add_n(all_fmean) + fmean0
        fvar0 = Knn if full_cov else Knn[..., None] - tf.add_n(all_fvar0)
        fvar = tf.add_n(all_fvar) + fvar0
        return fmean, _expand_independent_outputs(fvar, full_cov, full_output_cov)


def none_sum(*args):
    subset = [a for a in args if a is not None]
    if len(subset) == 0:
        return None
    else:
        return tf.add_n(subset)
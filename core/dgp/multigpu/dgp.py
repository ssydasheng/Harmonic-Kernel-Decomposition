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

import tensorflow as tf

from gpflow import params_as_tensors, name_scope
from gpflow import settings


from core.utils.utils import none_sum
from ..dgp import DGP


class MultiGPU_DGP(DGP):
    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        return tf.constant(0., dtype=settings.float_type)

    @name_scope('likelihood')
    @params_as_tensors
    def build_objective_and_grads(self):
        L, samples, dists = self.E_log_p_Y_with_intermediate(self.X, self.Y)

        L = tf.reduce_sum(L)
        self.elbo_kl = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, settings.float_type)
        scale /= tf.cast(tf.shape(self.X)[0], settings.float_type)  # minibatch size
        self.elbo_logll = L * scale
        elbo = self.elbo_logll - self.elbo_kl

        all_KLs =  sum([l.KLs for l in self.layers], [])
        like_vars = tf.trainable_variables(self.likelihood.name)
        vars = list(set(tf.trainable_variables()) - set(like_vars))

        ### setup for computing gradients
        grad_dist_like_kl_ = tf.gradients(-elbo, dists[-1].params_after_detach + like_vars + all_KLs)

        grad_output_dist = grad_dist_like_kl_[:len(dists[-1].params_after_detach)]
        grad_like = grad_dist_like_kl_[len(dists[-1].params_after_detach): -len(all_KLs)]
        grad_kls = grad_dist_like_kl_[-len(all_KLs):]
        grad_layerwise_kls = []
        n = 0
        for l in self.layers:
            grad_layerwise_kls.append(grad_kls[n: n+len(l.KLs)])
            n = n + len(l.KLs)

        ### backprop through layers
        all_grads_params = []
        samples = [None] + samples
        grad_params, grad_input = self.layers[-1].backprop(vars, samples[-2], None, grad_output_dist, grad_layerwise_kls[-1])
        all_grads_params.append(grad_params)
        for layer, layer_input, grad_KLs in zip(self.layers[:-1][::-1], samples[:-2][::-1], grad_layerwise_kls[:-1][::-1]):
            grad, grad_input = layer.backprop(vars, layer_input, grad_input, None, grad_KLs)
            all_grads_params.append(grad)

        grad_vars = [none_sum(*a) for a in zip(*all_grads_params)] + grad_like
        grad_and_vars = [(g, v) for g, v in zip(grad_vars, vars + like_vars)]
        return elbo, grad_and_vars

    @name_scope('ElogpY')
    def E_log_p_Y_with_intermediate(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
        with MC samples
        """
        S = self.num_samples
        full_cov = False
        original_Fs, dists = self._propagate(X, full_cov, S)
        num_outputs = [l.num_outputs for l in self.layers]
        Fs = [tf.reshape(F, [S, tf.shape(X)[0], no]) for no, F in zip(num_outputs, original_Fs)]
        Fmeans = [tf.reshape(dist.mean, [S, tf.shape(X)[0], no]) for no, dist in zip(num_outputs, dists)]
        Fvars = [tf.reshape(dist.var, [S, tf.shape(X)[0], no]) for no, dist in zip(num_outputs, dists)]

        if self.integrate_likelihood:
            var_exp = self.likelihood.variational_expectations(Fmeans[-1], Fvars[-1], Y, X)  # S, N, D
        else:
            var_exp = self.likelihood.logp(Fs[-1], Y, X)
        return tf.reduce_mean(var_exp, 0), original_Fs, dists  # N, D

    @params_as_tensors
    def _propagate(self, X, full_cov=False, S=1):
        Fs, dists = [], []
        F = tf.tile(tf.expand_dims(X, 0), [S, 1, 1]) if full_cov else tf.tile(X, [S, 1])

        for layer in self.layers:
            F, dist = layer.sample_from_conditional(F, full_cov=full_cov)

            Fs.append(F)
            dists.append(dist)
        return Fs, dists

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1):
        Fs, dists = self._propagate(X, full_cov, S)
        if not full_cov:
            num_outputs = [l.num_outputs for l in self.layers]
            Fs = [tf.reshape(F, [S, tf.shape(X)[0], no]) for no, F in zip(num_outputs, Fs)]
            Fmeans = [tf.reshape(dist.mean, [S, tf.shape(X)[0], no]) for no, dist in zip(num_outputs, dists)]
            Fvars = [tf.reshape(dist.var, [S, tf.shape(X)[0], no]) for no, dist in zip(num_outputs, dists)]
        else:
            # [S, N, D], [S, N, D, D]
            Fmeans = [dist.mean for dist in dists]
            Fvars = [dist.cov for dist in dists]
        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1):
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[-1], Fvars[-1]

    @name_scope('ElogpY')
    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
        with MC samples
        """
        Fs, Fmeans, Fvars = self.propagate(X, full_cov=False, S=self.num_samples)
        if self.integrate_likelihood:
            var_exp = self.likelihood.variational_expectations(Fmeans[-1], Fvars[-1], Y, X)  # S, N, D
        else:
            var_exp = self.likelihood.logp(Fs[-1], Y, X)
        return tf.reduce_mean(var_exp, 0)  # N, D

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        L = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))
        self.elbo_kl = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, settings.float_type)
        scale /= tf.cast(tf.shape(self.X)[0], settings.float_type)  # minibatch size
        self.elbo_logll = L * scale
        return self.elbo_logll - self.elbo_kl

    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar, Xnew)

    def predict_density(self, Xnew, Ynew, num_samples):
        if self.integrate_likelihood:
            Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
            l = self.likelihood.predict_density(Fmean, Fvar, Ynew, Xnew)
        else:
            Fs, _, _ = self.propagate(Xnew, full_cov=False, S=num_samples)
            l = self.likelihood.logp(Fs[-1], Ynew, Xnew)
        log_num_samples = tf.log(tf.cast(num_samples, settings.float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

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

from gpflow.params import DataHolder, Minibatch
from gpflow import params_as_tensors, ParamList, name_scope
from gpflow.models.model import Model
from gpflow import settings


from .utils.utils import BroadcastingLikelihood


class DGP(Model):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.

    """

    def __init__(self, X, Y, layers, likelihood,
                 minibatch_size=None, num_latent=None,
                 num_samples=1, num_data=None, XY_tensor=False,
                 integrate_likelihood=True, **kwargs):
        Model.__init__(self, **kwargs)
        self.num_samples = num_samples
        self.num_data = num_data or X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.integrate_likelihood = integrate_likelihood
        if XY_tensor:
            self.X = X
            self.Y = Y
        elif minibatch_size:
            self.X = Minibatch(X, minibatch_size, seed=0)
            self.Y = Minibatch(Y, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

        self.likelihood = BroadcastingLikelihood(likelihood)
        self.layers = ParamList(layers)

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

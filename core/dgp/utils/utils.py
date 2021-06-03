# Copyright 2017 Hugh Salimbeni
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
import numpy as np

from gpflow import settings
from gpflow.params import Parameter
from gpflow import transforms
from gpflow import params_as_tensors, Parameterized
from gpflow.likelihoods import Gaussian, Likelihood, Bernoulli, MultiClass
from gpflow import logdensities


def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise

    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + settings.jitter) ** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None, None, :, :] # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND


class BroadcastingLikelihood(Parameterized):
    """
    A wrapper for the likelihood to broadcast over the samples dimension. The Gaussian doesn't
    need this, but for the others we can apply reshaping and tiling.

    With this wrapper all likelihood functions behave correctly with inputs of shape S,N,D,
    but with Y still of shape N,D
    """
    def __init__(self, likelihood):
        Parameterized.__init__(self)
        self.likelihood = likelihood

        if isinstance(likelihood, Gaussian):
            self.needs_broadcasting = False
        else:
            self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if self.needs_broadcasting is False:
            return f(vars_SND, [tf.expand_dims(v, 0) for v in vars_ND])

        else:
            S, N, D = [tf.shape(vars_SND[0])[i] for i in range(3)]
            vars_tiled = [tf.tile(x[None, :, :], [S, 1, 1]) for x in vars_ND]

            flattened_SND = [tf.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [tf.reshape(x, [S*N, -1]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [tf.reshape(x, [S, N, -1]) for x in flattened_result]
            else:
                return tf.reshape(flattened_result, [S, N, -1])

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y, *args):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0],
                                                                                vars_SND[1],
                                                                                vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])

    @params_as_tensors
    def logp(self, F, Y, *args):
        f = lambda vars_SND, vars_ND: self.likelihood.logp(vars_SND[0], vars_ND[0])
        return self._broadcast(f, [F], [Y])

    @params_as_tensors
    def conditional_mean(self, F):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(vars_SND[0])
         return self._broadcast(f,[F], [])

    @params_as_tensors
    def conditional_variance(self, F, *args):
         f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(vars_SND[0])
         return self._broadcast(f,[F], [])

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar, *args):
         f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(vars_SND[0],
                                                                             vars_SND[1])
         return self._broadcast(f,[Fmu, Fvar], [])

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y, *args):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_density(vars_SND[0],
                                                                       vars_SND[1],
                                                                       vars_ND[0])
        return self._broadcast(f,[Fmu, Fvar], [Y])


class HeteGaussian(Parameterized):
    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.net = net

    def _variance(self, X):
        return tf.nn.softplus(self.net.forward(X))

    @params_as_tensors
    def logp(self, F, Y, X):
        variance = self._variance(X)
        return logdensities.gaussian(Y, F, variance)

    @params_as_tensors
    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    @params_as_tensors
    def conditional_variance(self, F, X):
        raise NotImplementedError
        variance = self._variance(X)
        return tf.fill(tf.shape(F), tf.squeeze(variance))

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar, X):
        variance = self._variance(X)
        return tf.identity(Fmu), Fvar + variance

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y, X):
        variance = self._variance(X)
        return logdensities.gaussian(Y, Fmu, Fvar + variance)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y, X):
        variance = self._variance(X)
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / variance


def crossEntropyBernoulli(x, p):
    return x * tf.log(p) + (1.-x) * tf.log(1-p)

class CrossEntropyBernoulli(Bernoulli):

    def logp(self, F, Y):
        return crossEntropyBernoulli(Y, self.invlink(F))

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return crossEntropyBernoulli(Y, p)


class SoftmaxMultiClass(Likelihood):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def logp(self, F, Y):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.to_int32(tf.squeeze(Y,-1)), logits=F)[..., None]

    def variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def conditional_mean(self, F):
        return tf.nn.softmax(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)


class TransGaussian(Likelihood):
    def __init__(self, variance=1.0, trans_func=None, **kwargs):
        super().__init__(**kwargs)
        self.trans_func = trans_func if trans_func is not None else tf.identity
        self.variance = Parameter(
            variance, transform=transforms.positive, dtype=settings.float_type)

    @params_as_tensors
    def logp(self, F, Y):
        F = self.trans_func(F)
        return logdensities.gaussian(Y, F, self.variance)

    @params_as_tensors
    def conditional_mean(self, F):  # pylint: disable=R0201
        if self.trans_func == tf.identity:
            return tf.identity(F)
        raise NotImplementedError

    @params_as_tensors
    def conditional_variance(self, F):
        F = self.trans_func(F)
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        Fmu = self.trans_func(Fmu)
        if self.trans_func == tf.identity:
            return tf.identity(Fmu), Fvar + self.variance
        else:
            return tf.identity(Fmu), tf.ones_like(Fvar) * -1. #TODO: this is wrong, it should be super careful.

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y):
        if self.trans_func == tf.identity:
            return logdensities.gaussian(Y, Fmu, Fvar + self.variance)
        raise NotImplementedError

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        if self.trans_func == tf.identity:
            return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
                   - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance
        raise NotImplementedError


def block_diagonal(A, B):
    nA, nB = tf.shape(A)[0], tf.shape(B)[0]
    dtype = A.dtype

    A_zero = tf.concat([A, tf.zeros([nA, nB], dtype=dtype)], 1)
    zero_B = tf.concat([tf.zeros([nB, nA], dtype=dtype), B], 1)
    AB = tf.concat([A_zero, zero_B], 0)
    return AB


def image_rotate(img, degree=180):
    # img: [n, H, H, C]
    if degree == 180:
        return [img, tf.image.flip_left_right(tf.image.flip_up_down(img))]
    elif degree == 90:
        img1 = tf.image.rot90(img)
        img2 = tf.image.rot90(img1)
        img3 = tf.image.rot90(img2)
        return [img, img1, img2, img3]
    else:
        raise NotImplementedError


def image_flip(img, degree=180):
    # img: [n, H, H, C]
    if degree == 180:
        return [img, tf.image.flip_left_right(tf.image.flip_up_down(img))]
    elif degree == 90:
        img1 = tf.image.flip_left_right(img)
        img2 = tf.image.flip_up_down(img1)
        img3 = tf.image.flip_up_down(img)
        return [img, img1, img2, img3]
    else:
        raise NotImplementedError

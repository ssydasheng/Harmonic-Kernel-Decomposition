import numpy as np
import tensorflow as tf
import gpflow
from gpflow import DataHolder, Minibatch, settings
from gpflow.conditionals import Kuu, Kuf
from gpflow.models import GPModel
from gpflow import features, transforms, params_as_tensors, kullback_leiblers
from gpflow.params import Parameter, ParamList
from gpflow.conditionals import conditional, _expand_independent_outputs
from gpflow.decors import name_scope

from core.utils.kl import batch_gauss_kl
from core.utils.conditional import blk_conditional
from core.utils.utils import BatchLowerTriangular


class BlockSVGP(GPModel):
    """
    Kernel matrix is 2x2 block diagonal, and Covariance Q is also 2x2 block diagonal.
    """
    def __init__(self, X, Y, kern, likelihood, feat,
                 mean_function=None,
                 num_latent=None,
                 whiten=True,
                 minibatch_size=None,
                 num_data=None,
                 XY_tensor=False,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - feat. Kuu and Kuf are both a list, representing the block diagonal groups.
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if XY_tensor:
            num_latent = num_latent or Y.get_shape().as_list()[1]
            assert num_data is not None
        else:
            if minibatch_size is None:
                X = DataHolder(X)
                Y = DataHolder(Y)
            else:
                X = Minibatch(X, batch_size=minibatch_size, seed=0)
                Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.whiten = whiten
        self.feature = feat

        # init Kernel matrix
        self.Kus = Kuu(self.feature, self.kern, jitter=settings.jitter) # [J, M, M]
        self.J = self.Kus.get_shape().as_list()[0]
        self.Lus = tf.cholesky(self.Kus) # [J, M, M]

        # init variational parameters
        M = self.feature.shape(0)
        mu = np.zeros((self.J, M, self.num_latent))
        self.qs_mu = Parameter(mu, dtype=settings.float_type)  # J x M x P

        # self.unpacked_qs_sqrt = ParamList([])
        # if not self.whiten:
        #     np_Lus = self.enquire_session().run(self.Lus)
        # for j in range(self.J):
        #     if self.whiten:
        #         sqrt = np.array([np.eye(M, dtype=settings.float_type) for _ in range(self.num_latent)])
        #     else:
        #         sqrt = np.tile(np_Lus[j][None, :, :], [self.num_latent, 1, 1])
        #     sqrt = Parameter(sqrt, transform=transforms.LowerTriangular(M, self.num_latent)) # P x M x M
        #     self.unpacked_qs_sqrt.append(sqrt)

        if not self.whiten:
            np_Lus = self.enquire_session().run(self.Lus).astype(settings.float_type)
        else:
            np_Lus = np.tile(np.eye(M, dtype=settings.float_type)[None], [self.J, 1, 1])
        np_Lus = np.tile(np_Lus[:, None], [1, num_latent, 1, 1])
        self.qs_sqrt = Parameter(np_Lus, transform=BatchLowerTriangular(M))

    # @property
    # @params_as_tensors
    # def qs_sqrt(self):
    #     return tf.stack([a.constrained_tensor for a in self.unpacked_qs_sqrt.params])

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            KL = batch_gauss_kl(self.qs_mu, self.qs_sqrt)
        else:
            KL = batch_gauss_kl(self.qs_mu, self.qs_sqrt, K_cholesky=self.Lus)
        return KL

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        N = tf.cast(self.num_data, settings.float_type)
        batch_size = tf.cast(tf.shape(self.X)[0], settings.float_type)
        scale = N / batch_size

        likelihood = tf.reduce_sum(var_exp) * scale - KL

        self.lld, self.kl = tf.reduce_sum(var_exp) / batch_size, KL / N
        return likelihood

    @params_as_tensors
    def _build_predict_separation(self, Xnew, full_cov=False):
        Kmsn = Kuf(self.feature, self.kern, Xnew)  # J x M x N
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)

        fmeans, fvar0, fvars = blk_conditional(
            Knn,
            Kmsn, self.Kus, self.Lus, self.qs_mu, self.qs_sqrt,
            full_cov=full_cov, white=self.whiten)
        fmean0 = self.mean_function(Xnew)
        return fmean0, fmeans, fvar0, fvars

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        fmean0, fmeans, fvar0, fvars = self._build_predict_separation(Xnew, full_cov=full_cov)
        dtype = Xnew.dtype
        mu = tf.cast(fmean0, dtype) + tf.cast(tf.reduce_sum(fmeans, 0), dtype)
        var = tf.cast(fvar0, dtype) + tf.cast(tf.reduce_sum(fvars, 0), dtype)
        return mu, _expand_independent_outputs(var, full_cov, full_output_cov)


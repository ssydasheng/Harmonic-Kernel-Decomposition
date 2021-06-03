# This code is adapted from the GPflow repository.
import gpflow
import numpy as np
import tensorflow as tf
from gpflow import DataHolder, Minibatch, settings
from gpflow import features, transforms, params_as_tensors, kullback_leiblers
from gpflow.conditionals import Kuu, Kuf
from gpflow.conditionals import _expand_independent_outputs
from gpflow.models import GPModel
from gpflow.params import Parameter


class SVGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """
    def __init__(self, X, Y, kern, likelihood, feat=None,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 double_whiten=False,
                 minibatch_size=None,
                 Z=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None,
                 kl_weight=1.,
                 p_diag=False,
                 XY_tensor=False,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        - p_diag is a boolean. If True, the Kuu is diagonal.
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
        self.q_diag, self.whiten, self.double_whiten = q_diag, whiten, double_whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.kl_weight = kl_weight
        self.p_diag = p_diag

        # init Kernel matrix
        self.Ku = Kuu(self.feature, self.kern, jitter=settings.jitter)
        if self.p_diag:
            self.Lu = self.Ku ** 0.5
        else:
            self.Lu = tf.cholesky(self.Ku)

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)
        if q_sqrt is None and not self.q_diag and not self.whiten:
            Lu = self.enquire_session().run(self.Lu)
            self.q_sqrt = np.tile(Lu[None, :, :], [self.num_latent, 1, 1])
        if q_sqrt is None and self.p_diag and not self.whiten and self.q_diag:
            Lu = self.enquire_session().run(self.Lu)
            self.q_sqrt = np.tile(Lu[:, None], [1, self.num_latent])
        if q_sqrt is None and self.p_diag and not self.whiten and not self.q_diag:
            Lu = self.enquire_session().run(self.Lu)
            self.q_sqrt = np.tile(np.diag(Lu)[None, :, :], [self.num_latent, 1, 1])

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """

        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(
                    q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(
                    q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.p_diag and self.whiten:
            if self.double_whiten:
                q_mu = self.Lu[..., None] * self.q_mu
            else:
                q_mu = self.q_mu
            return gpflow.kullback_leiblers.gauss_kl(q_mu, self.q_sqrt)

        if self.p_diag and not self.whiten:
            if self.double_whiten:
                q_mu = self.Ku[..., None] * self.q_mu
            else:
                q_mu = self.q_mu
            return gpflow.kullback_leiblers.gauss_kl(q_mu, self.q_sqrt, K_cholesky=tf.matrix_diag(self.Lu))

        if not self.p_diag and self.whiten:
            if self.double_whiten:
                q_mu = tf.matmul(self.Lu, self.q_mu, transpose_a=True)
            else:
                q_mu = self.q_mu
            return gpflow.kullback_leiblers.gauss_kl(q_mu, self.q_sqrt)

        if not self.p_diag and not self.whiten:
            if self.double_whiten:
                q_mu = tf.matmul(self.Ku, self.q_mu)
            else:
                q_mu = self.q_mu
            return gpflow.kullback_leiblers.gauss_kl(q_mu, self.q_sqrt, K_cholesky=self.Lu)

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

        likelihood = tf.reduce_sum(var_exp) * scale - KL * self.kl_weight

        self.lld, self.kl = tf.reduce_sum(var_exp) / batch_size, KL / N
        return likelihood

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        Kmn = Kuf(self.feature, self.kern, Xnew)  # M x N
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)


        mu, var = base_conditional(Kmn, self.Ku, Knn, self.q_mu, Lm=self.Lu, q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten, double_white=self.double_whiten, p_diag=self.p_diag)
        return mu + self.mean_function(Xnew), _expand_independent_outputs(var, full_cov, full_output_cov)



def base_conditional(Kmn, Kmm, Knn, f, *, Lm=None, full_cov=False,
                     q_sqrt=None, white=False, double_white=False, p_diag=False):
    r"""
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: M x [...] x N
    :param Kmm: M x M
    :param Knn: [...] x N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :param p_diag: bool
    :return: N x R  or R x N x N
    """
    # compute kernel stuff
    num_func = tf.shape(f)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f)[-2]

    # get the leadings dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat([tf.reshape(tf.range(1, K-1), [K-2]), # leading dims (...)
                      tf.reshape(0, [1]),  # [M]
                      tf.reshape(K-1, [1])], 0)  # [N]
    Kmn = tf.transpose(Kmn, perm)  # ... x M x N

    leading_dims = tf.shape(Kmn)[:-2]
    if Lm is None:
        if p_diag:
            Lm = Kmm ** 0.5 # [M]
        else:
            Lm = tf.cholesky(Kmm)  # [M,M]

    # Compute the projection matrix A
    if p_diag:
        A = Kmn / Lm[..., None] # [...,M,N]
    else:
        Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [...,M,M]
        A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # [...,M,N]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)  # [...,N,N]
        cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -3), cov_shape)  # [...,R,N,N]
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [...,N]
        cov_shape = tf.concat([leading_dims, [num_func, N]], 0) # [...,R,N]
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [...,R,N]

    # another backsubstitution in the unwhitened case
    if not white:
        if p_diag:
            A = A / Lm[..., None]
        else:
            A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [...,M,R]
    f = tf.broadcast_to(f, f_shape)  # [...,M,R]

    if double_white:
        fmean = tf.matmul(Kmn, f, transpose_a=True)
    else:
        fmean = tf.matmul(A, f, transpose_a=True)  # [...,N,R]

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = q_sqrt
            L = tf.broadcast_to(L, tf.concat([leading_dims, tf.shape(L)], 0))

            shape = tf.concat([leading_dims, [num_func, M, N]], 0)
            A_tiled = tf.broadcast_to(tf.expand_dims(A, -3), shape)
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # R x N

    if not full_cov:
        fvar = tf.matrix_transpose(fvar)  # N x R

    return fmean, fvar  # N x R, R x N x N or N x R

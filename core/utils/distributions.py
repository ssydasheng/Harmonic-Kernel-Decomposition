import tensorflow as tf
import tensorflow_probability as tfp

from gpflow import settings


class GaussianDistribution:

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def var(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

class UnivariateGaussian(GaussianDistribution):
    def __init__(self, mean, var):
        self._mean = mean
        self._var = var

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return  self._var

    def sample(self):
        eps = tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype)
        return self.mean + eps * tf.sqrt(self.var)


class SumGaussian(GaussianDistribution):
    def __init__(self, gausses):
        self._gausses = gausses

    @property
    def mean(self):
        return tf.add_n([g.mean for g in self._gausses])

    @property
    def var(self):
        return tf.add_n([g.var for g in self._gausses])

    @property
    def cov(self):
        return tf.add_n([g.cov for g in self._gausses])

    def sample(self):
        return tf.add_n([g.sample() for g in self._gausses])

    @property
    def params_before_detach(self):
        return sum([g.params_before_detach for g in self._gausses], [])

    @property
    def params_after_detach(self):
        return sum([g.params_after_detach for g in self._gausses], [])


class RowIndMVG(GaussianDistribution):
    """
    This class represents the summation of J independent Gaussians.

    :param mean: [J, N, R]
    :param var_u: [J, N]
    :param cov_v: [J, R, R]
    :param cov_v_sqrt: [J, R, R]
    """
    def __init__(self, mean, var_u, cov_v_sqrt):
        self._mean = mean
        self.J, self.N, self.R  = tf.shape(mean)[-3], tf.shape(mean)[-2], tf.shape(mean)[-1]
        self.var_u = var_u
        self.cov_v = tf.matmul(cov_v_sqrt, cov_v_sqrt, transpose_b=True)
        self.cov_v_sqrt = cov_v_sqrt

    @property
    def mean(self): # [N, R]
        return tf.reduce_sum(self._mean, 0)

    @property
    def var(self): # [N, R]
        var_v = tf.matrix_diag_part(self.cov_v)[:, None, :]
        return tf.reduce_sum(self.var_u[..., None] * var_v, 0)

    def sample(self): # [N, R]
        eps = tf.random.normal(tf.shape(self._mean), dtype=self._mean.dtype)
        eps_v = tf.matmul(eps, self.cov_v_sqrt, transpose_b=True)
        res = self._mean + tf.sqrt(self.var_u[..., None]+settings.jitter) * eps_v
        return tf.reduce_sum(res, 0)


class LeadingDimRowIndMVG2(RowIndMVG):
    """
    This class represents the summation of J independent Gaussians.

    :param mean: [P, J, N, R]
    :param var_u: [P, J, N]
    :param cov_v: [J, R, R]
    :param cov_v_sqrt: [J, R, R]
    """
    def __init__(self, mean, var_u, cov_v_sqrt):
        super().__init__(mean, var_u, cov_v_sqrt)
        self.P = mean.get_shape().as_list()[0]
        self.R = mean.get_shape().as_list()[-1]

    def _reshape(self, val): # [N, PR]
        val = tf.transpose(val, [1, 0, 2])
        return tf.reshape(val, [tf.shape(self._mean)[-2], self.P * self.R])

    @property
    def mean(self):
        res = tf.reduce_mean(self._mean, 1)
        return self._reshape(res)

    @property
    def var(self):
        var_v = tf.matrix_diag_part(self.cov_v)[:, None, :]
        res = tf.reduce_sum(self.var_u[..., None] * var_v, 1)
        return self._reshape(res)

    def sample(self):
        eps = tf.random.normal(tf.shape(self._mean), dtype=self._mean.dtype)
        eps_v = tf.matmul(eps, self.cov_v_sqrt, transpose_b=True)
        res = self._mean + tf.sqrt(self.var_u[..., None] + settings.jitter) * eps_v
        res = tf.reduce_sum(res, 1)
        return self._reshape(res)


class MGRowIndMVG(RowIndMVG):
    def __init__(self, mean, var_u, cov_v_sqrt):
        self.params_before_detach = [mean, var_u, cov_v_sqrt]
        mean, var_u, cov_v_sqrt = tf.stop_gradient(mean), tf.stop_gradient(var_u), tf.stop_gradient(cov_v_sqrt)
        self.params_after_detach = [mean, var_u, cov_v_sqrt]
        super().__init__(mean, var_u, cov_v_sqrt)


class MGLeadingDimRowIndMVG2(LeadingDimRowIndMVG2):
    def __init__(self, mean, var_u, cov_v_sqrt):
        self.params_before_detach = [mean, var_u, cov_v_sqrt]
        mean, var_u, cov_v_sqrt = tf.stop_gradient(mean), tf.stop_gradient(var_u), tf.stop_gradient(cov_v_sqrt)
        self.params_after_detach = [mean, var_u, cov_v_sqrt]
        super().__init__(mean, var_u, cov_v_sqrt)
import os.path as osp
import sys
# sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

import tensorflow as tf
import numpy as np
import kfac
import tensorflow_probability as tfp

from core.utils.kl import compute_mvg_kl_divergence, compute_mvg_kl_divergence_whiten

class TestDistributions(tf.test.TestCase):
    def test_mvg_kl(self, N=3, d=2):
        mean1 = np.random.normal(size=[N, d]) / d
        mean_vec1 = np.reshape(mean1.transpose(), [-1])
        u_rand1 = tf.random_normal(shape=[N, N], dtype=tf.float64) / N
        v_rand1 = tf.random_normal(shape=[d, d], dtype=tf.float64) / d

        u_rand1 = tf.linalg.band_part(u_rand1, -1, 0)
        v_rand1 = tf.linalg.band_part(v_rand1, -1, 0)
        u1, v1 = tf.matmul(u_rand1, u_rand1, transpose_b=True), tf.matmul(v_rand1, v_rand1, transpose_b=True)

        mean2 = np.random.normal(size=[N, d]) / d
        mean_vec2 = np.reshape(mean2.transpose(), [-1])
        u_rand2 = tf.random_normal(shape=[N, N], dtype=tf.float64) / N
        v_rand2 = tf.random_normal(shape=[d, d], dtype=tf.float64) / d

        u_rand2 = tf.linalg.band_part(u_rand2, -1, 0)
        v_rand2 = tf.linalg.band_part(v_rand2, -1, 0)
        u2, v2 = tf.matmul(u_rand2, u_rand2, transpose_b=True), tf.matmul(v_rand2, v_rand2, transpose_b=True)

        cov1 = kfac.utils.kronecker_product(v1, u1)
        cov2 = kfac.utils.kronecker_product(v2, u2)
        dist_refer1 = tfp.distributions.MultivariateNormalFullCovariance(mean_vec1, cov1)
        dist_refer2 = tfp.distributions.MultivariateNormalFullCovariance(mean_vec2, cov2)

        u1_sqrt, v1_sqrt = u_rand1, v_rand1
        u2_sqrt, v2_sqrt = u_rand2, v_rand2
        ########################################## ##########################################
        #####################       test KL divergence between MVG     #####################
        ########################################## ##########################################

        kl = compute_mvg_kl_divergence((mean1, u1, v1), (mean2, u2, v2), jitter=0.)
        kl2 = compute_mvg_kl_divergence((mean1, u1_sqrt, v1_sqrt), (mean2, u2, v2), jitter=0.,
                                        sqrt_u1=True, sqrt_v1=True)
        kl3 = compute_mvg_kl_divergence((mean1, u1, v1), (mean2, u2_sqrt, v2_sqrt), jitter=0.,
                                        sqrt_u2=True, sqrt_v2=True)
        kl4 = compute_mvg_kl_divergence((mean1, u1_sqrt, v1_sqrt), (mean2, u2_sqrt, v2_sqrt), jitter=0.,
                                        sqrt_u1=True, sqrt_v1=True, sqrt_u2=True, sqrt_v2=True)
        kl_refer = tfp.distributions.kl_divergence(dist_refer1, dist_refer2)

        kl_self = compute_mvg_kl_divergence((mean1, u1, v1), (mean1, u1, v1), jitter=0.)
        kl_self_refer = tfp.distributions.kl_divergence(dist_refer1, dist_refer1)

        with self.session() as sess:
            for idx in range(10):
                print(idx)
                a, a2, a3, a4, b, c, d = sess.run([kl, kl2, kl3, kl4, kl_refer, kl_self_refer, kl_self])
                self.assertAllClose(a, b)
                self.assertAllClose(a2, b)
                self.assertAllClose(a3, b)
                self.assertAllClose(a4, b)
                self.assertAllClose(c, 0)
                self.assertAllClose(d, 0.)

    def test_batch_mvg_kl(self, J=2, N=3, d=2):
        mean1 = np.random.normal(size=[J, N, d]) / d
        mean_vec1 = np.reshape(mean1.transpose([0, 2, 1]), [J, -1])
        u_rand1 = tf.random_normal(shape=[J, N, N], dtype=tf.float64) / N
        v_rand1 = tf.random_normal(shape=[J, d, d], dtype=tf.float64) / d

        u_rand1 = tf.linalg.band_part(u_rand1, -1, 0)
        v_rand1 = tf.linalg.band_part(v_rand1, -1, 0)
        u1, v1 = tf.matmul(u_rand1, u_rand1, transpose_b=True), tf.matmul(v_rand1, v_rand1, transpose_b=True)

        mean2 = np.random.normal(size=[J, N, d]) / d
        mean_vec2 = np.reshape(mean2.transpose([0, 2, 1]), [J, -1])
        u_rand2 = tf.random_normal(shape=[J, N, N], dtype=tf.float64) / N
        v_rand2 = tf.random_normal(shape=[J, d, d], dtype=tf.float64) / d

        u_rand2 = tf.linalg.band_part(u_rand2, -1, 0)
        v_rand2 = tf.linalg.band_part(v_rand2, -1, 0)
        u2, v2 = tf.matmul(u_rand2, u_rand2, transpose_b=True), tf.matmul(v_rand2, v_rand2, transpose_b=True)

        cov1s, cov2s = [], []
        for j in range(J):
            cov1s.append(kfac.utils.kronecker_product(v1[j], u1[j]))
            cov2s.append(kfac.utils.kronecker_product(v2[j], u2[j]))
        cov1, cov2 = tf.stack(cov1s), tf.stack(cov2s)
        dist_refer1 = tfp.distributions.MultivariateNormalFullCovariance(mean_vec1, cov1)
        dist_refer2 = tfp.distributions.MultivariateNormalFullCovariance(mean_vec2, cov2)

        # u1_sqrt, v1_sqrt = tf.cholesky(u1), tf.cholesky(v1)
        # u2_sqrt, v2_sqrt = tf.cholesky(u2), tf.cholesky(v2)
        u1_sqrt, v1_sqrt = u_rand1, v_rand1
        u2_sqrt, v2_sqrt = u_rand2, v_rand2
        ########################################## ##########################################
        #####################       test KL divergence between MVG     #####################
        ########################################## ##########################################

        kl = compute_mvg_kl_divergence((mean1, u1, v1), (mean2, u2, v2), jitter=0.)
        kl2 = compute_mvg_kl_divergence((mean1, u1_sqrt, v1_sqrt), (mean2, u2, v2), jitter=0.,
                                        sqrt_u1=True, sqrt_v1=True)

        kl3 = compute_mvg_kl_divergence((mean1, u1, v1), (mean2, u2_sqrt, v2_sqrt), jitter=0.,
                                        sqrt_u2=True, sqrt_v2=True)
        kl4 = compute_mvg_kl_divergence((mean1, u1_sqrt, v1_sqrt), (mean2, u2_sqrt, v2_sqrt), jitter=0.,
                                        sqrt_u1=True, sqrt_v1=True, sqrt_u2=True, sqrt_v2=True)
        kl_refer = tf.reduce_sum(tfp.distributions.kl_divergence(dist_refer1, dist_refer2))

        kl_self = compute_mvg_kl_divergence((mean1, u1, v1), (mean1, u1, v1), jitter=0.)
        kl_self_refer = tf.reduce_sum(tfp.distributions.kl_divergence(dist_refer1, dist_refer1))

        with self.session() as sess:
            for idx in range(10):
                print(idx)
                a, a2, a3, a4, b, c, d = sess.run([kl, kl2, kl3, kl4, kl_refer, kl_self_refer, kl_self])
                self.assertAllClose(a, b)
                self.assertAllClose(a2, b)
                self.assertAllClose(a3, b)
                self.assertAllClose(a4, b)
                self.assertAllClose(c, 0)
                self.assertAllClose(d, 0.)


    def test_batch_mvg_kl_whiten(self, J=2, N=3, d=2):
        mean1 = np.random.normal(size=[J, N, d]) / d
        mean_vec1 = np.reshape(mean1.transpose([0, 2, 1]), [J, -1])
        u_rand1 = tf.random_normal(shape=[J, N, N], dtype=tf.float64) / N**0.5
        v_rand1 = tf.random_normal(shape=[J, d, d], dtype=tf.float64) / d**0.5

        u_rand1 = tf.linalg.band_part(u_rand1, -1, 0)
        v_rand1 = tf.linalg.band_part(v_rand1, -1, 0)
        u1, v1 = tf.matmul(u_rand1, u_rand1, transpose_b=True), tf.matmul(v_rand1, v_rand1, transpose_b=True)

        mean2 = np.zeros(shape=[J, N, d])
        mean_vec2 = np.reshape(mean2.transpose([0, 2, 1]), [J, -1])
        u_rand2 = tf.tile(tf.eye(N, dtype=tf.float64)[None], [J, 1, 1])
        v_rand2 = tf.tile(tf.eye(d, dtype=tf.float64)[None], [J, 1, 1])

        u_rand2 = tf.linalg.band_part(u_rand2, -1, 0)
        v_rand2 = tf.linalg.band_part(v_rand2, -1, 0)
        u2, v2 = tf.matmul(u_rand2, u_rand2, transpose_b=True), tf.matmul(v_rand2, v_rand2, transpose_b=True)

        cov1s, cov2s = [], []
        for j in range(J):
            cov1s.append(kfac.utils.kronecker_product(v1[j], u1[j]))
            cov2s.append(kfac.utils.kronecker_product(v2[j], u2[j]))
        cov1, cov2 = tf.stack(cov1s), tf.stack(cov2s)
        dist_refer1 = tfp.distributions.MultivariateNormalFullCovariance(mean_vec1, cov1)
        dist_refer2 = tfp.distributions.MultivariateNormalFullCovariance(mean_vec2, cov2)

        # u1_sqrt, v1_sqrt = tf.cholesky(u1), tf.cholesky(v1)
        # u2_sqrt, v2_sqrt = tf.cholesky(u2), tf.cholesky(v2)
        u1_sqrt, v1_sqrt = u_rand1, v_rand1
        u2_sqrt, v2_sqrt = u_rand2, v_rand2
        ########################################## ##########################################
        #####################       test KL divergence between MVG     #####################
        ########################################## ##########################################

        kl = compute_mvg_kl_divergence((mean1, u1, v1), (mean2, u2, v2), jitter=0.)
        kl_refer = tf.reduce_sum(tfp.distributions.kl_divergence(dist_refer1, dist_refer2))

        kl_whiten = compute_mvg_kl_divergence_whiten((mean1, u1_sqrt, v1_sqrt), jitter=0., sqrt_u1=True, sqrt_v1=True)

        with self.session() as sess:
            for idx in range(10):
                print(idx)
                a, b, c = sess.run([kl, kl_refer, kl_whiten])
                self.assertAllClose(c, b)
                self.assertAllClose(c, a)


    def test_rig_sample(self):
        N, d = 3, 2
        mean1 = np.random.normal(size=[N, d])
        u_rand1, v_rand1 = tf.random_normal(shape=[N], dtype=tf.float64), tf.random_normal(shape=[d, d],
                                                                                           dtype=tf.float64)
        u1, v1 = tf.square(u_rand1), tf.matmul(v_rand1, v_rand1, transpose_b=True)

        cov1 = kfac.utils.kronecker_product(v1, tf.matrix_diag(u1))

        S = 1000000
        mean2 = tf.tile(mean1[None, ...], [S, 1, 1])
        mvg = RowIndependentMVG(mean2, var_u=u1, cov_v=v1)
        samples = mvg.sample()
        mean_est = tf.reduce_mean(samples, 0)
        cov_est = tfp.stats.covariance(tf.reshape(tf.transpose(samples, [0, 2, 1]), [S, N * d]))

        with self.session() as sess:
            for _ in range(10):
                b, c, d = sess.run([cov1, mean_est, cov_est])
                self.assertAllClose(mean1, c, atol=1e-2, rtol=1e-2)
                self.assertAllClose(b, d, atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
    # TestDistributions().test_mvg_kl()
    # TestDistributions().test_mvg_diag_kl()
    TestDistributions().test_mvg_kl(5, 2)
    TestDistributions().test_mvg_kl(2, 5)
    TestDistributions().test_batch_mvg_kl(2, 2, 5)
    TestDistributions().test_batch_mvg_kl(2, 5, 2)
    TestDistributions().test_batch_mvg_kl_whiten(2, 2, 5)
    TestDistributions().test_batch_mvg_kl_whiten(2, 5, 2)
#     TestDistributions().test_summvg_mvg_kl()
#     TestDistributions().test_prec_summvg_mvg_kl()
#     TestDistributions().test_upper_prec_summvg_mvg_kl()
#     TestDistributions().test_mvg_sample()
#     TestDistributions().test_rig_sample()
#     TestDistributions().test_diag_mvg_var_sample()
#     TestDistributions().test_eigmvg_stdnormal_kl()
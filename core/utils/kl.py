import tensorflow as tf
import gpflow
from gpflow import settings


def batch_gauss_kl(q_mu, q_sqrt, K_cholesky=None):
    # q_mu: J x M x B
    # q_sqrt: J x B x M x M
    # K_cholesky: J x M x M

    white = (K_cholesky is None)

    J, M, B = tf.shape(q_mu)[0], tf.shape(q_mu)[1], tf.shape(q_mu)[2]

    if white:
        alpha = q_mu  # [J, M, B]
    else:
        if K_cholesky is not None:
            Lp = K_cholesky  # [J, M, M]
        # if K_cholesky_inv is None:
        #     id = tf.tile(tf.eye(M, dtype=settings.float_type)[None], [J, 1, 1])
        #     K_cholesky_inv = tf.matrix_triangular_solve(K_cholesky, id, lower=True)

        # alpha = tf.matmul(K_cholesky_inv, q_mu)  # [J, M, B]
        alpha = tf.matrix_triangular_solve(Lp, q_mu, lower=True)  # [J, M, B]

    Lq = Lq_full = tf.matrix_band_part(q_sqrt, -1, 0)  # force lower triangle # [J, B, M, M]
    Lq_diag = tf.matrix_diag_part(Lq)  # [J, M, B]

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - J * B * M
    constant = - tf.cast(tf.size(q_mu, out_type=tf.int64), dtype=settings.float_type)

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)))


    # Trace term: tr(Σp⁻¹ Σq)
    if white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        # K_cholesky_inv_full = tf.tile(tf.expand_dims(K_cholesky_inv, 1), [1, B, 1, 1]) # [J, B, M, M]
        # LpiLq = tf.matmul(K_cholesky_inv_full, Lq_full)
        Lp_full = tf.tile(tf.expand_dims(Lp, 1), [1, B, 1, 1]) # [J, B, M, M]
        LpiLq = tf.matrix_triangular_solve(Lp_full, Lq_full, lower=True)
        trace = tf.reduce_sum(tf.square(LpiLq)) # [J, B, M, M]

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not white:
        log_sqdiag_Lp = tf.log(tf.square(tf.matrix_diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        # If K is B x M x M, num_latent is no longer implicit, no need to multiply the single kernel logdet
        scale = tf.cast(B, settings.float_type)
        twoKL += scale * sum_log_sqdiag_Lp

    return 0.5 * twoKL




def compute_mvg_kl_divergence(param1, param2, jitter=1e-8,
                              sqrt_u1=False, sqrt_v1=False, sqrt_u2=False, sqrt_v2=False,
                              lower_u1=False, lower_v1=False):
    mean1, u1_or_sqrt_u1, v1_or_sqrt_v1 = param1
    mean2, u2_or_sqrt_u2, v2_or_sqrt_v2 = param2
    n, m = tf.shape(mean1)[-2], tf.shape(mean1)[-1]

    jitter_u = tf.eye(n, dtype=mean1.dtype) * jitter
    jitter_v = tf.eye(m, dtype=mean1.dtype) * jitter

    if sqrt_u1 is False:
        u1 = u1_or_sqrt_u1 + jitter_u
        u1_tril = tf.cholesky(u1 + jitter_u)
        lower_u1 = True
    else:
        u1_tril = u1_or_sqrt_u1
        u1 = tf.matmul(u1_tril, u1_tril, transpose_b=True)
    if sqrt_v1 is False:
        v1 = v1_or_sqrt_v1 + jitter_v
        v1_tril = tf.cholesky(v1 + jitter_v)
        lower_v1 = True
    else:
        v1_tril = v1_or_sqrt_v1
        v1 = tf.matmul(v1_tril, v1_tril, transpose_b=True)
    if sqrt_u2 is False:
        u2 = u2_or_sqrt_u2 + jitter_u
        u2_tril = tf.cholesky(u2 + jitter_u)
    else:
        u2_tril = u2_or_sqrt_u2
        u2 = tf.matmul(u2_tril, u2_tril, transpose_b=True)
    if sqrt_v2 is False:
        v2 = v2_or_sqrt_v2 + jitter_v
        v2_tril = tf.cholesky(v2 + jitter_v)
    else:
        v2_tril = v2_or_sqrt_v2
        v2 = tf.matmul(v2_tril, v2_tril, transpose_b=True)

    n, m = tf.cast(n, mean1.dtype), tf.cast(m, mean1.dtype)
    u1_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u1_tril))), -1) * 2.
    v1_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v1_tril))), -1) * 2.
    logdet_1 = u1_logdet + v1_logdet

    u2_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u2_tril))), -1) * 2.
    v2_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v2_tril))), -1) * 2.
    logdet_2 = u2_logdet + v2_logdet

    logdet_difference = logdet_2 - logdet_1
    const = tf.cast(n * m, mean1.dtype)

    def efficent_cho_solve(tril, mat):
        m, n = mat.get_shape().as_list()[-2:]
        if n <= m:
            return tf.cholesky_solve(tril, mat)

        eye = tf.eye(m, dtype=tril.dtype)
        if len(tril.get_shape().as_list()) == 3:
            eye = tf.tile(eye[None], [tf.shape(tril)[0], 1, 1])
        elif  len(tril.get_shape().as_list()) > 3:
            raise NotImplementedError
        tril_tril_inv = tf.cholesky_solve(tril, eye)
        return tril_tril_inv @ mat

    vec = mean1-mean2
    inverse_cov2_vec = efficent_cho_solve(u2_tril, tf.matrix_transpose(efficent_cho_solve(v2_tril, tf.matrix_transpose(vec))))
    # inverse_cov2_vec = tf.cholesky_solve(u2_tril, tf.matrix_transpose(tf.cholesky_solve(v2_tril, tf.matrix_transpose(vec))))
    mean_diff = tf.reduce_sum(vec * inverse_cov2_vec, [-1, -2])

    # trace = tf.trace(tf.cholesky_solve(v2_tril, v1)) * tf.trace(tf.cholesky_solve(u2_tril, u1))
    trace1 = tf.reduce_sum(tf.square(tf.matrix_triangular_solve(v2_tril, v1_tril)), [-1, -2])
    trace2 = tf.reduce_sum(tf.square(tf.matrix_triangular_solve(u2_tril, u1_tril)), [-1, -2])
    trace = trace1 * trace2

    kl = 0.5 * (logdet_difference - const + trace + mean_diff)
    return tf.reduce_sum(kl)


def compute_mvg_kl_divergence_whiten(param1, jitter=1e-8,
                                     sqrt_u1=False, sqrt_v1=False):
    mean1, u1_or_sqrt_u1, v1_or_sqrt_v1 = param1
    n, m = tf.shape(mean1)[-2], tf.shape(mean1)[-1]

    jitter_u = tf.eye(n, dtype=mean1.dtype) * jitter
    jitter_v = tf.eye(m, dtype=mean1.dtype) * jitter

    if sqrt_u1 is False:
        u1 = u1_or_sqrt_u1 + jitter_u
        u1_tril = tf.cholesky(u1 + jitter_u)
    else:
        u1_tril = u1_or_sqrt_u1
        u1 = tf.matmul(u1_tril, u1_tril, transpose_b=True)
    if sqrt_v1 is False:
        v1 = v1_or_sqrt_v1 + jitter_v
        v1_tril = tf.cholesky(v1 + jitter_v)
    else:
        v1_tril = v1_or_sqrt_v1
        v1 = tf.matmul(v1_tril, v1_tril, transpose_b=True)

    n, m = tf.cast(n, mean1.dtype), tf.cast(m, mean1.dtype)
    u1_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u1_tril))), -1) * 2.
    v1_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v1_tril))), -1) * 2.
    logdet_1 = u1_logdet + v1_logdet

    logdet_2 = 0.

    logdet_difference = logdet_2 - logdet_1
    const = tf.cast(n * m, mean1.dtype)

    def efficent_cho_solve(tril, mat):
        m, n = mat.get_shape().as_list()[-2:]
        if n <= m:
            return tf.cholesky_solve(tril, mat)

        eye = tf.eye(m, dtype=tril.dtype)
        if len(tril.get_shape().as_list()) == 3:
            eye = tf.tile(eye[None], [tf.shape(tril)[0], 1, 1])
        elif  len(tril.get_shape().as_list()) > 3:
            raise NotImplementedError
        tril_tril_inv = tf.cholesky_solve(tril, eye)
        return tril_tril_inv @ mat

    mean_diff = tf.reduce_sum(mean1**2., [-1, -2])

    trace1 = tf.reduce_sum(tf.square(v1_tril), [-1, -2])
    trace2 = tf.reduce_sum(tf.square(u1_tril), [-1, -2])
    trace = trace1 * trace2

    kl = 0.5 * (logdet_difference - const + trace + mean_diff)
    return tf.reduce_sum(kl)


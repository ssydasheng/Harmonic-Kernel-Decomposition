
import tensorflow as tf
from gpflow import settings



def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False, Lm=None):
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
        Lm = tf.cholesky(Kmm)  # [M,M]

    # Compute the projection matrix A
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
        A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [...,M,R]
    f = tf.broadcast_to(f, f_shape)  # [...,M,R]
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



def blk_conditional(Knn, Kmsn, Kms, Lms, qs_mu, qs_sqrt,
                    full_cov=False, white=False, return_nystrom_residual=True):
    r"""
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Knn: N x N  or  N
    :param Kmsn: J x M x N
    :param Kms: J x M x M
    :param Lms: J x M x M
    :param qs_mu: J x M x R
    :param qs_sqrt: J x R x M x M
    :param full_cov: bool
    :param white: bool
    """
    # compute kernel stuff
    Knn = tf.cast(Knn, Lms.dtype) if Knn is not None else Knn
    Kmsn = tf.cast(Kmsn, Lms.dtype)
    qs_mu = tf.cast(qs_mu, Lms.dtype)
    qs_sqrt = tf.cast(qs_sqrt, Lms.dtype)

    num_func = tf.shape(qs_mu)[-1]  # R
    N = tf.shape(Kmsn)[-1]
    if len(Lms.get_shape().as_list())==3:
        batch_J = True
    elif len(Lms.get_shape().as_list())==2:
        batch_J = False
    else:
        raise NotImplementedError

    # Compute the projection matrix A
    As = tf.matrix_triangular_solve(Lms, Kmsn, lower=True) # [J, M, N]

    # compute the covariance due to the conditioning

    if return_nystrom_residual:
        if full_cov:
            if batch_J:
                fvar0 = Knn - tf.reduce_sum(tf.matmul(As, As, transpose_a=True), 0) # [N, N]
            else:
                fvar0 = Knn - tf.matmul(As, As, transpose_a=True) # [N, N]
            fvar0 = tf.broadcast_to(tf.expand_dims(fvar0, -3), [num_func, N, N])  # [R,N,N]
        else:
            if batch_J:
                fvar0 = Knn - tf.reduce_sum(tf.reduce_sum(tf.square(As), -2), 0)  # [N]
            else:
                fvar0 = Knn - tf.reduce_sum(tf.square(As), -2) # [N]
            fvar0 = tf.broadcast_to(tf.expand_dims(fvar0, -2), [num_func, N])  # [R,N]
    else:
        if full_cov:
            if batch_J:
                fvar0 = tf.reduce_sum(tf.matmul(As, As, transpose_a=True), 0) # [N, N]
            else:
                fvar0 = tf.matmul(As, As, transpose_a=True)  # [N, N]
            fvar0 = tf.broadcast_to(tf.expand_dims(fvar0, -3), [num_func, N, N])  # [R,N,N]
        else:
            if batch_J:
                fvar0 = tf.reduce_sum(tf.reduce_sum(tf.square(As), -2), 0)  # [N]
            else:
                fvar0 = tf.reduce_sum(tf.square(As), -2)  # [N]
            fvar0 = tf.broadcast_to(tf.expand_dims(fvar0, -2), [num_func, N])  # [R,N]

    # another backsubstitution in the unwhitened case
    if not white:
        # As = tf.matmul(Lms_inv, As, transpose_a=True) # [J, M, N]
        # with tf.device('/cpu:0'):
        As = tf.matrix_triangular_solve(tf.matrix_transpose(Lms), As, lower=False) # [J, M, N]

    fmeans = tf.matmul(As, qs_mu, transpose_a=True) # [J, N, R]

    # if qs_sqrt[0].get_shape().ndims == 2:
    #     raise NotImplementedError
    # elif qs_sqrt[0].get_shape().ndims == 3:
    if batch_J:
        As_tiled = tf.tile(As[:, None], [1, num_func, 1, 1]) # J x R x M x N
    else:
        As_tiled = tf.tile(As[None], [num_func, 1, 1]) # R x M x N
    LTAs = tf.matmul(qs_sqrt, As_tiled, transpose_a=True)  # J x R x M x N
    # else:  # pragma: no cover
    #     raise ValueError("Bad dimension for q_sqrt: %s" %
    #                      str(qs_sqrt.get_shape().ndims))

    if full_cov:
        fvars = tf.matmul(LTAs, LTAs, transpose_a=True) # J x R x N x N
    else:
        fvars = tf.reduce_sum(tf.square(LTAs), -2) # J x R x N

    if not full_cov:
        fvar0 = tf.matrix_transpose(fvar0)  # J x N x R
        fvars = tf.matrix_transpose(fvars)  # J x N x R

    return fmeans, fvar0, fvars


def blk_mvg_conditional(Kmsn, Kms, Knn, qs_mu, cov_us_sqrt, full_cov=False, white=False, Lms=None,
                        return_nystrom_residual=True):
    """
    P might be empty.

    :param Kmsn: [P, J, M, N]
    :param Kms: [J, M, M]
    :param Knn: P x N x N  or  P x N
    :param Lms: [J, M, M]
    :param qs_mu: [J, M, R]
    :param cov_us_sqrt: [J, M, M]
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool

    :return fmean: [P, J, N, R]
    :return para_u_cov: [P, J, N] or [P, J, N, N]
    :return orth_u_cov: [P, N] or [P, N, N]
    """
    if len(Kms.get_shape().as_list()) == 2:
        batch_J = False
    elif len(Kms.get_shape().as_list()) == 3:
        batch_J = True
        J = Kms.get_shape().as_list()[0]
    else:
        raise NotImplementedError

    if len(Kmsn.get_shape().as_list()) > len(Kms.get_shape().as_list()):
        has_leading_dim = True
    else:
        has_leading_dim = False

    N = tf.shape(Kmsn)[-1]
    M, R = Kmsn.get_shape().as_list()[-2], qs_mu.get_shape().as_list()[-1]
    float_type = settings.float_type


    if has_leading_dim:
        # firstly compute the inverse
        # [J, M, M]
        if batch_J:
            eye = tf.tile(tf.eye(M, dtype=float_type)[None], [J, 1, 1])
        else:
            eye = tf.eye(M, dtype=float_type)
        Lms_inv = tf.matrix_triangular_solve(Lms, eye, lower=True)

        # [P, J, M, N]
        A = Lms_inv @ Kmsn
        invKuu_Kuf = tf.matmul(Lms_inv, A, transpose_a=True)
        # tilded_Lms = tf.tile(Lms[None], [tf.shape(Kmsn)[0]] + [1] * (3 if batch_J else 2))
    else:
        tilded_Lms = Lms
        # [J, M, N]
        A = tf.matrix_triangular_solve(tilded_Lms, Kmsn, lower=True)
        invKuu_Kuf = tf.matrix_triangular_solve(tf.matrix_transpose(tilded_Lms), A, lower=False)

    if not white:
        A = invKuu_Kuf

    # [P, J, N, R]
    fmean = tf.matmul(A, qs_mu, transpose_a=True)

    # [J, M, M]
    cov_us = tf.matmul(cov_us_sqrt, cov_us_sqrt, transpose_b=True)
    para_B = tf.matmul(cov_us, A) # [P, J, M, N]

    if return_nystrom_residual:
        if full_cov:
            para_u_cov = tf.matmul(A, para_B, transpose_a=True) # [P, J, N, N]
            if batch_J:
                orth_u_cov = Knn - tf.reduce_sum(tf.matmul(Kmsn, invKuu_Kuf, transpose_a=True), -3)
            else:
                orth_u_cov = Knn - tf.matmul(Kmsn, invKuu_Kuf, transpose_a=True)
        else:
            para_u_cov = tf.reduce_sum(A * para_B, -2) # [P, J, N]
            if batch_J:
                orth_u_cov = Knn - tf.reduce_sum(tf.reduce_sum(invKuu_Kuf * Kmsn, -2), -2) # [P, N]
            else:
                orth_u_cov = Knn - tf.reduce_sum(invKuu_Kuf * Kmsn, -2) # [P, N]
    else:
        if full_cov:
            para_u_cov = tf.matmul(A, para_B, transpose_a=True) # [P, J, N, N]
            if batch_J:
                orth_u_cov = tf.reduce_sum(tf.matmul(Kmsn, invKuu_Kuf, transpose_a=True), -3)
            else:
                orth_u_cov = tf.matmul(Kmsn, invKuu_Kuf, transpose_a=True)
        else:
            para_u_cov = tf.reduce_sum(A * para_B, -2) # [P, J, N]
            if batch_J:
                orth_u_cov = tf.reduce_sum(tf.reduce_sum(invKuu_Kuf * Kmsn, -2), -2) # [P, N]
            else:
                orth_u_cov = tf.reduce_sum(invKuu_Kuf * Kmsn, -2) # [P, N]

    return fmean, para_u_cov, orth_u_cov

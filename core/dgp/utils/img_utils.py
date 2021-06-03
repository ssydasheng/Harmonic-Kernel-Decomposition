import numpy as np
import tensorflow as tf
from gpflow import settings
from sklearn import cluster
float_type = settings.float_type
from gpflow.session_manager import get_default_session

from .meanFunctions import identity_conv2d, xavier_conv2d


def _sample(tensor, count):
    chosen_indices = np.random.choice(np.arange(tensor.shape[0]), count)
    return tensor[chosen_indices]

##################### Sample Patches with the Corresponding Location ###################

def _sample_patches_with_locations(HW_image, N, patch_size, patch_length):
    out = np.zeros((N, patch_length+2))
    for i in range(N):
        patch_y = np.random.randint(0, HW_image.shape[0] - patch_size)
        patch_x = np.random.randint(0, HW_image.shape[1] - patch_size)
        out[i, :-2] = HW_image[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size].reshape(patch_length)
        out[i][-2] = patch_y
        out[i][-1] = patch_x
    return out

def _cluster_patches_with_locations(NHWC_X, M, patch_size):
    NHWC = NHWC_X.shape
    patch_length = patch_size ** 2 * NHWC[3]
    # Randomly sample images and patches.
    patches_per_image = 1
    samples = min(10000, 100 * M)
    # samples = min(50000, 100 * M)
    patches = np.zeros((samples, patch_length+2), dtype=settings.float_type)
    for i in range(samples // patches_per_image):
        # Sample a random image, compute the patches and sample some random patches.
        image = _sample(NHWC_X, 1)[0]
        sampled_patches = _sample_patches_with_locations(
            image, patches_per_image, patch_size, patch_length)
        patches[i*patches_per_image:(i+1)*patches_per_image] = sampled_patches

    k_means = cluster.KMeans(n_clusters=M, init='random', n_jobs=-1)
    k_means.fit(patches)
    z = k_means.cluster_centers_
    return z


##################### Compute Euclidean distances using convolution ###################

def kerne_by_conv(NHWC_X, Z, view, base_kern, return_patch_square=True):
    N = tf.shape(NHWC_X)[0]
    filter_size = [view.filter_size, view.filter_size, tf.shape(NHWC_X)[3]]
    ls = base_kern.lengthscales
    if base_kern.ARD:
        weights_filter = tf.reshape(1. / ls ** 2., filter_size + [1])
    else:
        weights_filter = tf.ones(filter_size + [1], dtype=NHWC_X.dtype) / ls ** 2.
    z_filter = tf.reshape(tf.transpose(Z / ls ** 2.), filter_size + [tf.shape(Z)[0]])

    if return_patch_square:
        # [batch_size, output_size_0, output_size_1, 1]
        patch_square = tf.nn.conv2d(NHWC_X ** 2., filter=weights_filter,
                                    padding=view.padding, strides=view.stride)
    else:
        patch_square = None
    # [batch_size, output_size_0, output_size_1, num_inducing]
    patch_times_z = tf.nn.conv2d(NHWC_X, filter=z_filter, padding=view.padding, strides=view.stride)
    # [num_inducing]
    z_square = tf.reduce_sum(Z ** 2. / ls ** 2., -1)

    return patch_square, z_square, patch_times_z


##################### Used for Initializing Conv Layers ###################

def image_HW(patch_count):
    image_height = int(np.sqrt(patch_count))
    return [image_height, image_height]

def identity_conv(NHWC_X, filter_size, feature_maps_in, feature_maps_out, stride, padding="VALID"):
    random_images = np.random.choice(np.arange(NHWC_X.shape[0]), size=1000)
    sess = tf.get_default_session()
    if sess is None:
        sess = get_default_session()
    return sess.run(identity_conv2d(NHWC_X[random_images], filter_size, feature_maps_in, feature_maps_out,
                                    stride, padding=padding))

def xavier_conv(NHWC_X, filter_size, feature_maps_in, feature_maps_out, stride, padding="VALID"):
    random_images = np.random.choice(np.arange(NHWC_X.shape[0]), size=1000)
    sess = tf.get_default_session()
    if sess is None:
        sess = get_default_session()
    return sess.run(xavier_conv2d(NHWC_X[random_images], filter_size, feature_maps_in, feature_maps_out,
                                  stride, padding=padding))

def select_initial_inducing_points(X, M):
    if X.shape[0] > 2000:
        random_images = np.random.choice(np.arange(X.shape[0]), size=2000)
        X = X[random_images]
    kmeans = cluster.KMeans(n_clusters=M, init='k-means++', n_jobs=-1)
    kmeans.fit(X)
    return kmeans.cluster_centers_

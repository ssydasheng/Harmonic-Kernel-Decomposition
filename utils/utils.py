import os
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import tensorflow as tf
import shutil
from functools import wraps



def default_plotting_new():
   plt.rcParams['font.size'] = 15
   plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
   plt.rcParams['axes.titlesize'] = 1.2 * plt.rcParams['font.size']
   plt.rcParams['legend.fontsize'] = 1.0 * plt.rcParams['font.size']
   plt.rcParams['xtick.labelsize'] = 1.0 * plt.rcParams['font.size']
   plt.rcParams['ytick.labelsize'] = 1.0 * plt.rcParams['font.size']
   plt.rcParams['axes.ymargin'] = 0
   plt.rcParams['axes.xmargin'] = 0


def default_plotting():
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.5 * plt.rcParams['font.size']
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.xmargin'] = 0


def merge_dicts(*dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


def get_kemans_init(x, k_centers):
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]

    if k_centers > 1000:
        return x[:k_centers]

    kmeans = MiniBatchKMeans(n_clusters=k_centers, batch_size=k_centers*10).fit(x)
    return kmeans.cluster_centers_


def median_distance_global(x, nmax=10000, x2=None):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[1] > 50:
        nmax = 1000

    if x.shape[0] > nmax:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:nmax]]
    if x2 is None:
        x2 = x
    else:
        if x2.shape[0] > nmax:
            permutation = np.random.permutation(x2.shape[0])
            x2 = x2[permutation[:nmax]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x2, 0)
    dis_a = np.sqrt(np.sum((x_col - x_row) ** 2, -1)) # [n, n]
    return np.median(dis_a)


def median_distance_local(x, nmax=10000, x2=None):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[1] > 50:
        nmax = 1000

    if x.shape[0] > nmax:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:nmax]]
    if x2 is None:
        x2 = x
    else:
        if x2.shape[0] > nmax:
            permutation = np.random.permutation(x2.shape[0])
            x2 = x2[permutation[:nmax]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x2, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0) * (x.shape[1] ** 0.5)


def variational_expectations(Fmu, Fvar, Y, variance):
    return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(variance) \
           - 0.5 * (tf.square(Y - Fmu) + Fvar) / variance


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


def reuse_variables(scope):
    """
    From zhusuan (https://github.com/thu-ml/zhusuan/)
    A decorator for transparent reuse of tensorflow
    `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_ in a
    function. The decorated function will automatically create variables the
    first time they are called and reuse them thereafter.
    .. note::
        This decorator is internally implemented by tensorflow's
        :func:`make_template` function. See `its doc
        <https://www.tensorflow.org/api_docs/python/tf/make_template>`_
        for requirements on the target function.
    :param scope: A string. The scope name passed to tensorflow
        `variable_scope()
        <https://www.tensorflow.org/api_docs/python/tf/variable_scope>`_.
    """
    return lambda f: tf.make_template(scope, f)


def restore_model(args, print_func, saver, sess, MODEL_PATH):
    ckpt_file = tf.train.latest_checkpoint(MODEL_PATH)
    begin_epoch = 1
    if ckpt_file is not None:
        print_func('Restoring model from {}...'.format(ckpt_file))
        begin_epoch = int(ckpt_file.split('epoch.')[-1].split('.ckpt')[0])
        saver.restore(sess, ckpt_file)
    return begin_epoch

def save_model(args, print_func, saver, sess, MODEL_PATH, epoch):
    print_func('Saving model...')
    save_path = os.path.join(MODEL_PATH, "epoch.{}.ckpt".format(epoch))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    saver.save(sess, save_path)

def initialize_gpflow_graph(model):
    sess = model.enquire_session()
    sess.run(tf.global_variables_initializer(), feed_dict=model.initializable_feeds)
    return sess

def setup_summary(summary_path):
    summary_op = tf.summary.merge_all()
    if os.path.exists(summary_path):
        shutil.rmtree(summary_path)
    summary_writer = tf.summary.FileWriter(summary_path)
    return summary_op, summary_writer

def with_resource_factory(environment):
    def with_resource(f):
        @wraps
        def wrapper(self, *a, **kw):
            with environment:
                return f(self, *a, **kw)
        return wrapper
    return with_resource

def extend_dict_of_list(dict_, new_dict):
    for key, value in new_dict.items():
        dict_[key].append(value)
import os
import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_path)
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import gpflow
import numpy as np
import tensorflow as tf
from gpflow.features import Kuu, Kuf

from core.svgp.reflection import ReflectionFeatures
from utils.data import get_regression_data


NZ = 1024
NX = 3000
kern = 'matern32'

def median_distance_global(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.sqrt(np.sum((x_col - x_row) ** 2, -1)) # [n, n]
    return np.median(dis_a)

def get_data(dataset):
    data = get_regression_data(dataset, split=0)
    X = data.X_train
    np.random.seed(1234)
    perm = np.random.permutation(X.shape[0])
    X = X[perm[:NX+NZ]]
    return X[:NX], X[NX:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='wilson_3droad', type=str)
    parser.add_argument("--feature", default='ref2', type=str)
    parser.add_argument("--J", default=2, type=int)
    parser.add_argument("--m", type=int, default=2)
    args = parser.parse_args()
    np.random.seed(1234)

    ###################### load data ######################

    X, Z = get_data(args.dataset)
    Z = Z[:args.m]
    input_dim = X.shape[1]

    ls = median_distance_global(X)
    kern = gpflow.kernels.Matern32(input_dim, lengthscales=ls)

    with tf.variable_scope('feature'):
        if args.feature == 'svgp':
            feat = gpflow.features.InducingPoints(Z)
        elif args.feature == 'ref2':
            U = np.eye(Z.shape[-1]).astype('float64')
            feat = ReflectionFeatures([Z] * args.J, U, name='feature')
        elif args.feature == 'ref2pca':
            Sigma = np.cov(X, rowvar=False)
            Lambda, U = np.linalg.eigh(Sigma)
            feat = ReflectionFeatures([Z] * args.J, U, name='feature')
        else:
            raise NotImplementedError

    Kxx = kern.Kdiag(X)
    Kzx = Kuf(feat, kern, X) # [J, NZ, NX]
    Kzz = Kuu(feat, kern, jitter=1e-10)
    Lzz = tf.linalg.cholesky(Kzz) # [J, Z, Z]

    Lzz_inv_Kxz = tf.matrix_triangular_solve(Lzz, Kzx, lower=True)
    if len(Lzz_inv_Kxz.get_shape().as_list()) == 3:
        est_Kxx = tf.reduce_sum(Lzz_inv_Kxz**2., [0, 1])
    elif len(Lzz_inv_Kxz.get_shape().as_list()) == 2:
        est_Kxx = tf.reduce_sum(Lzz_inv_Kxz**2., [0])
    else:
        raise NotImplementedError
    error = tf.reduce_mean(Kxx - est_Kxx)

    sess = kern.enquire_session()
    feeds = {}
    feeds.update(kern.initializable_feeds)
    feeds.update(feat.initializable_feeds)
    sess.run(tf.global_variables_initializer(), feed_dict=feeds)

    if not osp.exists(osp.join(root_path, 'results/nystrom/')):
        os.makedirs(osp.join(root_path, 'results/nystrom/'))
    path = osp.join(root_path, 'results/nystrom/%s_%s_%d_m%d.npz' % (
        args.dataset, args.feature, args.J, args.m))
    with open(path, 'wb') as file:
        np.savez(file, error=sess.run(error))


def plot_dataset():

    def init_plotting():
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["figure.figsize"] = [12, 6]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 2.4 * plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 1.5 * plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 1.5 * plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 1.5 * plt.rcParams['font.size']
    init_plotting()

    Ms = [16, 32, 64, 128, 256, 512]
    Js = [4, 16]
    fig, axes = plt.subplots(1, 5, figsize=(36, 6), sharex=True)

    colors = ['#969696', '#252525']
    colors_ref2 = ['#a1d99b', '#31a354']
    colors_refpca = ['#9ecae1', '#3182bd']

    for idx, dataset in enumerate(['trueyear', 'wilson_3droad', 'wilson_song', 'wilson_buzz', 'wilson_houseelectric']):
        res_svgp_Ms = []
        for m in Ms:
            path = osp.join(root_path, 'results/nystrom/%s_%s_%d_m%d.npz' % (
                dataset, 'svgp', 0, m))
            res = np.load(path)
            res_svgp_Ms.append(res['error'])

        res_svgp_2Ms = []
        for m in Ms:
            path = osp.join(root_path, 'results/nystrom/%s_%s_%d_m%d.npz' % (
                dataset, 'svgp', 0, 2*m))
            res = np.load(path)
            res_svgp_2Ms.append(res['error'])

        res_ref2_JMs = []
        for J in Js:
            res_ref2_JMs.append([])
            for m in Ms:
                path = osp.join(root_path, 'results/nystrom/%s_%s_%d_m%d.npz' % (
                    dataset, 'ref2', J, m))
                res = np.load(path)
                res_ref2_JMs[-1].append(res['error'])

        res_refpca_JMs = []
        for J in Js:
            res_refpca_JMs.append([])
            for m in Ms:
                path = osp.join(root_path, 'results/nystrom/%s_%s_%d_m%d.npz' % (
                    dataset, 'ref2pca', J, m))
                res = np.load(path)
                res_refpca_JMs[-1].append(res['error'])

        axes[idx].plot(Ms, res_svgp_Ms, colors[0], lw=6., label=r'$M$')
        axes[idx].plot(Ms, res_svgp_2Ms, colors[1], lw=6., label=r'$2M$')
        for ii, (c, res) in enumerate(zip(colors_ref2, res_ref2_JMs)):
            axes[idx].plot(Ms, res, c, lw=6., label=r'$AXES: T \times M$' if ii==len(res_ref2_JMs)-1 else None)

        for ii, (c, res) in enumerate(zip(colors_refpca, res_refpca_JMs)):
            axes[idx].plot(Ms, res, c, lw=6., label=r'$PCA: T \times M$' if ii==len(res_refpca_JMs)-1 else None)

        title = "%s" % (dataset.split('_')[-1])
        if title == 'trueyear':
            title = 'year'
        axes[idx].set_title(title)
        axes[idx].set_yscale('log')

    axes[0].set_ylabel('Nystrom Residuals')

    for j in range(5):
        axes[j].set_xlabel('Inducing Points (M)')
        axes[j].set_xscale('log', basex=2)
        axes[j].set_xticks(Ms)
    axes[0].legend()

    plt.tight_layout()
    plt.savefig('figures/nystrom.pdf')


def bash():
    experiments = []
    for dataset in ['trueyear', 'wilson_3droad', 'wilson_song', 'wilson_buzz', 'wilson_houseelectric']:
        for m in [16, 32, 64, 128, 256, 512, 1024]:
            experiments.append(
                "python exp/nystrom_trace_error.py --dataset %s --kern %s --J %d --feature %s --m %d" % (
                    dataset, kern, 0, 'svgp', m))
            for feat in ['ref2', 'ref2pca']:
                for J in [4, 16]:
                    experiments.append("python exp/nystrom_trace_error.py --dataset %s --kern %s --J %d --feature %s --m %d" % (
                        dataset, kern, J, feat, m))

if __name__ == '__main__':
    bash()
    # main()
    # plot_dataset()
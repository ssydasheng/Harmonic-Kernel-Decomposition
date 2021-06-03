import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_path)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

import numpy as np

NX = 3000
kern = 'matern32'

def init_plotting():
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.7 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.2 * plt.rcParams['font.size']
init_plotting()

def DFT_MATRIX(J):
    JJt = np.arange(J).reshape([J, 1]) * np.arange(J).reshape([1, J])
    real = np.cos(2 * np.pi / J * JJt) / J
    img = np.sin(2 * np.pi / J * JJt) / J
    return real + img * 1j

def plot_contour(x,y,z,resolution = 50,contour_method='linear'):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(x):max(x):complex(resolution),   min(y):max(y):complex(resolution)]
    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method)
    return X,Y,Z

def rotate():
    np.random.seed(1234)
    a = 4
    N = 1000
    original_X = np.random.uniform(low=-a/2., high=a/2., size=[N, 2])
    DFT = DFT_MATRIX(4)

    ### define the $G$  for translation
    T = 4
    def G_rotate(x):
        return np.concatenate([-x[:, 1:], x[:, :1]], 1)

    def RBF(X1, X2):
        dist = np.sum((X1[:, None] - X2) ** 2., -1)
        return np.exp(-dist / 2.)

    X1 = original_X # map original x to [0. a]
    X2 = G_rotate(X1)
    X3 = G_rotate(X2)
    X4 = G_rotate(X3)
    X5 = G_rotate(X4)
    assert np.all(np.isclose(X5, X1))

    K11 = RBF(X1, X1)
    K12 = RBF(X1, X2)
    K13 = RBF(X1, X3)
    K14 = RBF(X1, X4)

    K1s = np.concatenate([K11[..., None], K12[..., None], K13[..., None], K14[..., None]], -1) @ DFT
    K1s = np.transpose(K1s, [2, 0, 1])
    chol = np.linalg.cholesky(K1s + 1e-8 * np.eye(N))
    random_f = chol @ np.random.normal(size=[4, N, 1])

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(16, 4))
    for j in range(4):
        x, y, z = plot_contour(original_X[:, 0], original_X[:, 1], np.real(random_f[j]).squeeze())
        axes[j].contour(x, y, z, cmap='viridis')
        axes[j].set_title(r'$t=%d$' % j)

    plt.tight_layout()
    plt.pause(1e-18)
    plt.savefig('figures/rotate.pdf')

if __name__ == '__main__':
    rotate()

import os
import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_path)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def init_plotting():
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.7 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.2 * plt.rcParams['font.size']
init_plotting()

batch_size = 256
learning_rate = 0.003
kern = 'matern32'
run = 'float64'


Ms = [1000]
criterions = ['valid_rmse_sq', 'valid_nll', 'time_per_iter']

names_dict = dict(valid_rmse_sq='valid_rmse_sq',
                 valid_nll='valid_lld',
                 time_per_iter='time_per_iter',
                 byte_in_use='byte_in_use',
                  train_loss='train_loss',
                  test_nll='test_lld',
                 test_rmse_sq = 'test_rmse_sq',
                  )
map_dict = dict(valid_rmse_sq=lambda t: t**0.5, valid_nll=lambda t: -t,
                test_rmse_sq=lambda t: t**0.5, test_nll=lambda t: -t,
                time_per_iter=lambda t: t, byte_in_use=lambda t: t,
                train_loss=lambda t:t)
criterion_name_dict = dict(valid_rmse_sq='VALID RMSE',
                 valid_nll='VALID NLL',
                 time_per_iter='time per iteration',
                 byte_in_use='byte_in_use',
                  train_loss='train loss',
                  test_nll='TEST NLL',
                 test_rmse_sq = 'TEST RMSE',
                  )

epoch_dict = dict(valid_rmse_sq=30, valid_nll=30, test_rmse_sq=30, test_nll=30,
                time_per_iter=300, byte_in_use=300, train_loss=300)

dataset_dict = dict(trueyear='year', wilson_3droad='3droad', wilson_song='song',
                    wilson_buzz='buzz', wilson_houseelectric='houseelectric')

datasets = ['trueyear', 'wilson_3droad', 'wilson_song', 'wilson_buzz' , 'wilson_houseelectric']

def load_res(criterion, dataset, model, J, M):
    alls = []
    if model == 'refmg' or model == 'ref2mg' or model == 'reframg':
        ng = J
    else:
        ng = 1

    if dataset == 'year' or dataset == 'trueyear':
        splits = [0]
    else:
        splits = [0, 1, 2]

    if dataset == 'trueyear' and kern == 'matern32':
        runrun = 'debug'
    else:
        runrun = run

    for split in splits:
        PATH = 'results/regression/{}/{}_{}_{}/J{}/bs{}_lr{}/NG{}_{}_split{}'.format(
        dataset, model, kern, M, J, batch_size, learning_rate, ng, runrun, split)

        if osp.exists(osp.join(root_path, PATH, 'res.npz')):
            with open(osp.join(root_path, PATH, 'res.npz'), 'rb') as file:
                res = map_dict[criterion](np.load(file)[names_dict[criterion]])
                if len(res) < epoch_dict[criterion]:
                    print(PATH, len(res))
                    res = np.concatenate([res[0] * np.ones([epoch_dict[criterion]-len(res)]), res], 0)
                alls.append(res)
        else:
            print(osp.join(root_path, PATH, 'res.npz'))
            exit()
    data = np.asarray(alls)
    return data.mean(0), data.std(0) / len(data)**0.5

def plot_inducing_dataset_criterion(ax, dataset, criterion):
    colors_ref2 = ['#a1d99b', '#31a354']
    colors_refpca = ['#9ecae1', '#3182bd']

    M = Ms[0]
    svgp_M = load_res(criterion, dataset, 'svgp', 0, M)
    svgp_2M = load_res(criterion, dataset, 'svgp', 0, 2*M)

    ref_J8 = load_res(criterion, dataset, 'refmg', 8, M)
    ref_J8_2M = load_res(criterion, dataset, 'refmg', 8, 2*M)


    min_ = min(svgp_M[0][-1], svgp_2M[0][-1], ref_J8[0][-1], ref_J8_2M[0][-1])
    max_ = max(svgp_M[0][-1], svgp_2M[0][-1], ref_J8[0][-1], ref_J8_2M[0][-1])
    new_min = min_ - 0.2 * (max_ - min_)
    new_max = max_ + 0.2 * (max_ - min_)

    ax.bar(1, svgp_M[0][-1], yerr=svgp_M[1][-1], color=colors_ref2[0])
    ax.bar(2, svgp_2M[0][-1], yerr=svgp_2M[1][-1], color=colors_ref2[1])
    ax.bar(3, ref_J8[0][-1], yerr=ref_J8[1][-1], color=colors_refpca[0])
    ax.bar(4, ref_J8_2M[0][-1], yerr=ref_J8_2M[1][-1], color=colors_refpca[1])

    ax.set_ylim([new_min, new_max])
    ax.set_xticks([])

def plot_inducing_dataset(axes, dataset):
    for ax, criterion in zip(axes, criterions):
        plot_inducing_dataset_criterion(ax, dataset, criterion)

def plot_training_dataset_criterion(ax, dataset, criterion, M, epoch_per_pt=1):

    colors_ref2 = ['#a1d99b', '#31a354']
    colors_refpca = ['#9ecae1', '#3182bd']
    colors = colors_ref2 + [None, None] + colors_refpca


    svgp_M = load_res(criterion, dataset, 'svgp', 0, M)[0]
    svgp_2M = load_res(criterion, dataset, 'svgp', 0, 2*M)[0]

    ref_J8 = load_res(criterion, dataset, 'refmg', 8, M)[0]
    ref_J8_2M = load_res(criterion, dataset, 'refmg', 8, 2*M)[0]
    epochs = np.arange(len(svgp_M)) * epoch_per_pt

    ax.plot(epochs, svgp_M, c=colors[0], lw=4, label=r'$M$')
    ax.plot(epochs, svgp_2M, c=colors[1], lw=4, label=r'$2M$')
    ax.plot(epochs, ref_J8, c=colors[4], lw=4, label=r'$8 \times M$ ')
    ax.plot(epochs, ref_J8_2M, c=colors[5], lw=4, label=r'$8 \times 2M$ ')

    min_ = min(svgp_M.tolist() + svgp_2M.tolist() + ref_J8.tolist() + ref_J8_2M.tolist())
    max_ = max(svgp_M[2:].tolist() + svgp_2M[2:].tolist() + ref_J8[2:].tolist() + ref_J8_2M[2:].tolist())

    if criterion == 'train_loss':
        new_min = min_ - 0.02 * (max_ - min_)
        new_max = min_ + 0.5 * (max_ - min_)
    else:
        new_min = min_ - 0.2 * (max_ - min_)
        new_max = max_ + 0.2 * (max_ - min_)
    ax.set_ylim([new_min, new_max])

def plot_nll(criterion):
    fig, axes = plt.subplots(1, len(datasets), figsize=[5 * len(datasets), 4], sharex=True)
    for idx, dataset in enumerate(datasets):
        plot_inducing_dataset_criterion(axes[idx], dataset, criterion)

    for i in range(len(datasets)):
        axes[i].set_title(dataset_dict[datasets[i]])

    axes[0].set_ylabel(criterion_name_dict[criterion])
    for i in range(len(datasets)):
        plt.sca(axes[i])
        plt.xticks([1, 2, 3, 4], [r'$M$', r'$2M$', r'$8xM$', r'$8x2M$'])
    plt.tight_layout()
    plt.savefig('figures/bar_M1K_inducing_%s-%s.pdf' % (criterion, kern))

def plot_training_and_time(dataset):
    def init_plotting():
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["figure.figsize"] = [12, 6]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 1.2 * plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 1.7 * plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 1. * plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 1.2 * plt.rcParams['font.size']

    init_plotting()

    M = Ms[0]
    fig, axes = plt.subplots(1, 2, figsize=[10, 4])
    plot_training_dataset_criterion(axes[0], dataset, 'test_nll', M, epoch_per_pt=10)
    plot_inducing_dataset_criterion(axes[1], dataset, 'time_per_iter')

    axes[0].set_ylabel(criterion_name_dict['test_nll'])
    axes[0].set_xlabel('epochs')
    axes[0].legend()

    axes[1].set_ylabel(criterion_name_dict['time_per_iter'])
    plt.sca(axes[1])
    plt.xticks([1, 2, 3, 4], [r'$M$', r'$2M$', r'$8xM$', r'$8x2M$'])

    plt.tight_layout()
    plt.savefig('figures/all_M1K_training-%s-%s.pdf' % (dataset, kern))


if __name__ == '__main__':
    plot_nll('test_nll')
    plot_training_and_time('wilson_3droad')

import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_path)

import time
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

import torch
import gpytorch

from utils.logging import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='kin8nm', nargs='?', type=str)
parser.add_argument("--split", default=0, type=int)

parser.add_argument("--m", type=int, default=20)
parser.add_argument("--iter_per_epoch", type=int, default=20)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.003)

parser.add_argument("-run", "--run", type=str, default='debug')
parser.add_argument("--ckpt-dir", type=str)
args = parser.parse_args()

str_ = 'elevation/ski/{}/m{}_lr{}_ipe{}/{}_split{}'.format(
    args.dataset, args.m, args.learning_rate, args.iter_per_epoch,
    args.run, args.split
)
logger = get_logger('ski', 'results/'+str_, __file__)
print = logger.info
print(args.__dict__)


def gen_dataset(args):
    with open(osp.join(root_path, 'data', 'ETOPO.npz'), 'rb') as file:
        data = np.load(file)
        longitude, latitude, elevation = data['longitude'], data['latitude'], data['height']
        X = np.concatenate([longitude[..., None], latitude[..., None]], axis=1)
        y = elevation[..., None]
    mean_y, std_y = elevation.mean(), elevation.std()
    y = (y - mean_y) / std_y
    N = longitude.shape[0]
    n_train = int(N * 0.8)
    perm = np.random.RandomState(0).permutation(longitude.shape[0])
    train_x, train_y = X[perm[:n_train]], y[perm[:n_train]]
    test_x, test_y = X[perm[n_train:]], y[perm[n_train:]]
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.1, random_state=np.random.RandomState(args.split))
    print('train_x {} | test_x {} | valid_x {}'.format(train_x.shape, test_x.shape, valid_x.shape))

    def coordinate(long_lati):
        long, lati = long_lati[..., 0], long_lati[..., 1]
        long = long / 180. * np.pi  # [-pi, pi]
        lati = lati / 180. * np.pi  # [-pi/2, pi/2]

        x = np.sin(np.pi / 2. - lati) * np.cos(long)
        y = np.sin(np.pi / 2. - lati) * np.sin(long)
        z = np.cos(np.pi / 2. - lati)
        return np.concatenate([x[..., None], y[..., None], z[..., None]], -1)
    train_x = coordinate(train_x)
    valid_x = coordinate(valid_x)
    test_x = coordinate(test_x)

    perm = np.random.RandomState(1).permutation(train_x.shape[0])
    train_x, train_y = train_x[perm[:600000]], train_y[perm[:600000]]

    max_x = np.maximum(np.maximum(train_x.max(0), valid_x.max(0)), test_x.max(0))
    min_x = np.minimum(np.minimum(train_x.min(0), valid_x.min(0)), test_x.min(0))
    train_x = (train_x - min_x) / (max_x - min_x) * 2. - 1.
    valid_x = (valid_x - min_x) / (max_x - min_x) * 2. - 1.
    test_x = (test_x - min_x) / (max_x - min_x) * 2. - 1. # TODO: map to 0-1
    return train_x, valid_x, test_x, train_y, valid_y, test_y, mean_y, std_y, X, y


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # SKI requires a grid size hyperparameter. This util can help with that
        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=3
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():
    train_x, valid_x, test_x, train_y, valid_y, test_y, mean_y, std_y, X, y = gen_dataset(args)
    train_x, train_y = torch.from_numpy(train_x).float().contiguous(), torch.from_numpy(train_y).float().contiguous()
    valid_x, valid_y = torch.from_numpy(valid_x).float().contiguous(), torch.from_numpy(valid_y).float().contiguous()
    test_x, test_y = torch.from_numpy(test_x).float().contiguous(), torch.from_numpy(test_y).float().contiguous()
    train_y, valid_y, test_y = train_y.squeeze(-1), valid_y.squeeze(-1), test_y.squeeze(-1)

    if torch.cuda.is_available():
        train_x, train_y = train_x.cuda(), train_y.cuda()
        valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.initialize(noise=0.1)
    model = GPRegressionModel(train_x, train_y, likelihood)
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    results_logs = {'train_loss': [], 'time_per_iter': [],
                    'valid_rmse_sq': [], 'valid_lld': [],
                    'test_rmse_sq': [], 'test_lld': []}

    iter_per_epoch = args.iter_per_epoch
    def train_epoch(epoch):
        model.train()
        likelihood.train()
        begin_time = time.time()
        losses = []
        for i in range(iter_per_epoch):
            optimizer.zero_grad()

            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            losses.append(loss.item())
            if i % 1 == 0:
                print('Epoch %d | Iter %d/%d - Loss: %.3f - nose : %.3f' % (epoch, i + 1, iter_per_epoch, loss.item(), likelihood.noise.item()))

            optimizer.step()
            torch.cuda.empty_cache()

        elapsed_time = time.time() - begin_time
        results_logs['train_loss'].append(np.mean(losses))
        results_logs['time_per_iter'].append(elapsed_time / iter_per_epoch)
        return np.mean(losses), elapsed_time / iter_per_epoch

    epochs = int(args.iterations / iter_per_epoch)
    args.epochs = epochs
    for epoch in range(args.epochs):
        with gpytorch.settings.max_cg_iterations(500):
            avg_loss, time_per_iter = train_epoch(epoch)

        model.eval()
        likelihood.eval()
        prev_time = time.time()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            rmse_sqs, llds = [], []
            batch_size = 50000
            n_batches = int(np.ceil(test_x.shape[0] / float(batch_size)))
            for i in range(n_batches):

                x_batch = test_x[i*batch_size: (i+1)*batch_size]
                y_batch = test_y[i*batch_size: (i+1)*batch_size]
                preds = model(x_batch)

                rmse_sqs.append((preds.mean - y_batch)**2.)
                llds.append(likelihood.log_marginal(y_batch, preds))
            rmse_sq = torch.cat(rmse_sqs,dim=0).mean().item()
            lld = torch.cat(llds, dim=0).mean().item()

            results_logs['test_rmse_sq'].append(rmse_sq)
            results_logs['test_lld'].append(lld)

        print('Test use {}'.format(time.time() - prev_time))
        print('Epoch [%d /%d] | %.3f time_per_iter | average loss = %.3f | valid rmse_sq = %.3f | valid llds = %.3f' % (
            epoch, args.epochs, time_per_iter, avg_loss, rmse_sq, lld) )

        with open(osp.join('results/', str_, 'res.npz'), 'wb') as f:
            np.savez(f, **results_logs)

if __name__ == '__main__':
    main()


import os.path as osp

import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

from gpflow.kernels import RBF
np.random.seed(8)

data_path = '/h/ssy/codes/data'


def load_toy(name):
    return dict(
        step=Step,
        nonHomo=NonHomo1,
        nonHomo2=NonHomo2,
        snelson=Snelson_toy,
        sin1=sin1_toy,
    )[name]


def gen_non_homogeneous():
    n = 40
    id = np.eye(n) * 1e-5

    n2 = 30
    id2 = np.eye(n2) * 1e-5

    x1 = np.random.uniform(-5.5, -3.5, size=[n, 1])
    rbf1 = RBF(1, lengthscales=0.3)
    cov1 = rbf1.enquire_session().run(rbf1.K(x1)) + id
    y1 = np.matmul(np.linalg.cholesky(cov1), np.random.normal(size=[n, 1]))

    x2 = np.random.uniform(-2., 1., size=[n2, 1])
    rbf2 = RBF(1, lengthscales=3)
    cov2 = rbf2.enquire_session().run(rbf2.K(x2)) + id2
    y2 = np.matmul(np.linalg.cholesky(cov2), np.random.normal(size=[n2, 1]))

    x3 = np.random.uniform(2.5, 4.5, size=[n, 1])
    rbf3 = RBF(1, lengthscales=0.3)
    cov3 = rbf3.enquire_session().run(rbf3.K(x3)) + id
    y3 = np.matmul(np.linalg.cholesky(cov3), np.random.normal(size=[n, 1]))

    xs = np.concatenate([x1, x2, x3], 0)
    ys = np.concatenate([y1, y2, y3], 0)
    plt.scatter(xs.squeeze(), ys.squeeze())
    plt.pause(1e-19)
    # plt.show(block=True)
    # plt.savefig('res.pdf')
    with open('nonHomo.npz', 'w') as f:
        np.savez(f, x=xs, y=ys)


class toy_dataset(object):
    def __init__(self, name=''):
        self.name = name

    def train_samples(self):
        raise NotImplementedError

    def test_samples(self):
        raise NotImplementedError

class gp_rbf_different_ls_toy(toy_dataset):
    def __init__(self, name='gp_rbf_different_ls'):
        self.x_min = -10
        self.x_max = 10
        self.y_min = -2
        self.y_max = 6
        self.confidence_coeff = 1.
        self.y_std = 0.2
        hparams = [
            {'name': 'RBF', 'params': {'lengthscales': 1.0, 'input_dim': 1, 'name': 'k0'}},
            {'name': 'RBF', 'params': {'lengthscales': 0.1, 'input_dim': 1, 'name': 'k1'}}
        ]
        self.kerns = [
            RBF(1, lengthscales=1.0),
            RBF(1, lengthscales=0.1)
        ]
        super(gp_rbf_different_ls_toy, self).__init__(name)
        self.sample()

    def sample(self, n_data=40, n_test=200):
        np.random.seed(1)

        X1_train_test = np.concatenate([np.random.uniform(self.x_min * 0.8, 0, (n_data // 2, 1)),
                                        np.linspace(self.x_min, 0, num=200, dtype=np.float32).reshape([n_test, 1])],
                                       0)
        X2_train_test = np.concatenate([np.random.uniform(0, self.x_max * 0.8, (n_data // 2, 1)),
                                        np.linspace(0, self.x_max, num=n_test, dtype=np.float32).reshape([200, 1])],
                                       0)
        epsilon = np.random.normal(0, self.y_std, (n_data,))
        cov1 = self.kerns[0].K(X1_train_test, X1_train_test) * 9
        cov2 = self.kerns[1].K(X2_train_test, X2_train_test)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cov1 = sess.run(cov1)
        cov2 = sess.run(cov2)
        sess.close()

        mean = np.zeros(n_data // 2 + n_test)
        Y1_train_test = np.random.multivariate_normal(mean, cov1)

        mean = np.zeros(n_data - n_data // 2 + n_test)
        Y2_train_test = np.random.multivariate_normal(mean, cov2)

        X_train = np.concatenate([X1_train_test[:n_data // 2], X2_train_test[:n_data // 2]], 0)
        y_train = np.concatenate([Y1_train_test[:n_data // 2], Y2_train_test[:n_data // 2]], 0)
        y_train = np.squeeze(y_train + epsilon)

        X_test = np.concatenate([X1_train_test[n_data // 2:], X2_train_test[n_data // 2:]], 0)
        y_test = np.concatenate([np.squeeze(Y1_train_test[n_data // 2:]), np.squeeze(Y2_train_test[n_data // 2:])],
                                0)

        self.X_train = X_train  # .astype(dtype=np.float32)
        self.y_train = y_train  # .astype(dtype=np.float32)

        self.X_test = X_test  # .astype(dtype=np.float32)
        self.y_test = y_test  # .astype(dtype=np.float32)

        indices = np.argsort(np.squeeze(self.X_test))
        self.X_test = self.X_test[indices, :]
        self.y_test = self.y_test[indices]

    def train_samples(self, n_data=20):
        return self.X_train, self.y_train

    def test_samples(self):
        return self.X_test, self.y_test

    def plot(self):
        import matplotlib.pyplot as plt
        X_train = np.squeeze(self.X_train)
        X_test = np.squeeze(self.X_test)

        y_train = self.y_train
        y_test = self.y_test

        indices = np.argsort(X_train)
        X_train = X_train[indices]
        y_train = y_train[indices]

        indices = np.argsort(X_test)
        X_test = X_test[indices]
        y_test = y_test[indices]

        plt.plot(X_train, y_train)
        plt.plot(X_test, y_test)
        plt.pause(1e-19)
        plt.show(block=True)


class x3_toy(toy_dataset):
    def __init__(self, name='x3'):
        self.x_min = -6
        self.x_max = 6
        self.y_min = -100
        self.y_max = 100
        self.confidence_coeff = 3.
        self.f = lambda x: np.power(x, 3)
        self.y_std = 3.
        super(x3_toy, self).__init__(name)

    def train_samples(self, n_data=20):
        np.random.seed(1)

        X_train = np.random.uniform(-4, 4, (n_data, 1))
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(X_train ** 3 + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(-6, 6, num=1000, dtype=np.float32).reshape([1000, 1])
        outputs = np.power(inputs, 3)
        return inputs, outputs

class x3_gap_toy(toy_dataset):
    def __init__(self, name='x3_gap'):
        self.x_min = -6
        self.x_max = 6
        self.y_min = -100
        self.y_max = 100
        self.confidence_coeff = 3.
        self.y_std = 3.
        self.f = lambda x: np.power(x, 3)
        super(x3_gap_toy, self).__init__(name)

    def train_samples(self, n_data=20):
        np.random.seed(1)

        X_train_1 = np.random.uniform(-4, -1, (n_data // 2, 1))
        X_train_2 = np.random.uniform(1, 4, (n_data // 2, 1))
        X_train = np.concatenate([X_train_1, X_train_2], axis=0)
        epsilon = np.random.normal(0, self.y_std, (n_data, 1))
        y_train = np.squeeze(X_train ** 3 + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(-6, 6, num=1000, dtype=np.float32).reshape([1000, 1])
        outputs = np.power(inputs, 3)
        return inputs, outputs

class sin_toy(toy_dataset):
    def __init__(self, name='sin'):
        self.x_min = -5
        self.x_max = 5
        self.y_min = -3.5
        self.y_max = 3.5
        self.confidence_coeff = 1.
        self.y_std = 2e-1

        def f(x):
            return 2 * np.sin(4 * x)

        self.f = f
        super(sin_toy, self).__init__(name)

    def train_samples(self):
        np.random.seed(3)

        X_train1 = np.random.uniform(-2, -0.5, (10, 1))
        X_train2 = np.random.uniform(0.5, 2, (10, 1))
        X_train = np.concatenate([X_train1, X_train2], axis=0)
        epsilon = np.random.normal(0, self.y_std, (20, 1))
        y_train = np.squeeze(self.f(X_train) + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(self.x_min, self.x_max, 1000).reshape([1000, 1])
        return inputs, self.f(inputs)

class sin1_toy(toy_dataset):
    def __init__(self, name='sin1'):
        self.x_min = -1.5
        self.x_max = 1.5
        self.y_min = -1.5
        self.y_max = 1.5
        self.confidence_coeff = 1.
        self.y_std = 2e-1

        def f(x):
            return np.sin(x * np.pi)

        self.f = f
        super(sin1_toy, self).__init__(name)

    def train_samples(self):
        np.random.seed(3)

        X_train = np.random.uniform(-1, 1, (50, 1))
        epsilon = np.random.normal(0, self.y_std, (50, 1))
        y_train = self.f(X_train) + epsilon
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(self.x_min, self.x_max, 200).reshape([200, 1])
        return inputs, self.f(inputs)

def load_snelson_data(n=100, dtype=np.float64, seed=1):
    if seed is not None:
        np.random.seed(seed)

    def _load_snelson(filename):
        with open(os.path.join(data_path, "snelson", filename), "r") as f:
            return np.array([float(i) for i in f.read().strip().split("\n")],
                            dtype=dtype)

    train_x = np.expand_dims(_load_snelson("train_inputs"), 1)
    train_y = _load_snelson("train_outputs")
    test_x = np.expand_dims(_load_snelson("test_inputs"), 1)
    perm = np.random.permutation(train_x.shape[0])
    train_x = train_x[perm][:n]
    train_y = train_y[perm][:n]
    return train_x, train_y, test_x

class Snelson_toy(toy_dataset):
    def __init__(self, n=100, name='snelson'):
        self.x_train, self.y_train, self.x_test = load_snelson_data(n)

        self.x_min = np.min(self.x_test)
        self.x_max = np.max(self.x_test)
        self.y_min = -5
        self.y_max = 5
        self.confidence_coeff = 3.
        self.y_std = None
        super(Snelson_toy, self).__init__(name)

    def train_samples(self):
        return self.x_train, self.y_train

    def test_samples(self):
        return self.x_test, np.zeros([self.x_test.shape[0]])


class Step(toy_dataset):
    def __init__(self, name='step'):
        np.random.seed(1)
        Ns = 300
        self.test_x = np.linspace(-5., 5., Ns)[:, None]

        N = 100
        self.train_x = np.random.normal(0, 1, N)[:, None]

        f_step = lambda x: 0. if x < 0 else 1.
        train_y = np.reshape([f_step(x) for x in self.train_x], self.train_x.shape)
        self.train_y = train_y + np.random.randn(*self.train_x.shape) * 1e-2

        self.x_min = np.min(self.test_x)
        self.x_max = np.max(self.test_x)
        self.y_min = -1.5
        self.y_max = 1.5
        self.confidence_coeff = 3.
        self.y_std = None
        super(Step, self).__init__(name)

    def train_samples(self):
        return self.train_x, self.train_y

    def test_samples(self):
        return self.test_x, np.zeros(self.test_x.shape)


class NonHomo1(toy_dataset):
    def __init__(self, name='nonHomo'):

        self.x_min = -8
        self.x_max = 8
        self.y_min = -1
        self.y_max = 1
        self.confidence_coeff = 1.
        self.y_std = None
        super(NonHomo1, self).__init__(name)

    def train_samples(self):
        res = np.load(osp.join(data_path, 'nonHomo1.npz'))
        return res['x'], res['y']

    def test_samples(self):
        return np.linspace(self.x_min, self.x_max, 200).reshape([200, 1]), np.zeros([200])

class NonHomo2(NonHomo1):
    def train_samples(self):
        res = np.load(osp.join(data_path, 'nonHomo2.npz'))
        return res['x'], res['y']

# if __name__ == '__main__':
#     gen_non_homogeneous()

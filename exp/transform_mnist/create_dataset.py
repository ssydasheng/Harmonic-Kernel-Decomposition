#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import os
import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

from scipy import ndimage
import numpy as np

from utils.dataset import load_mnist
from utils.img_utils import save_image_collections


def save_randomflip_mnist():
    train_x, train_y, test_x, test_y = load_mnist()

    np.random.seed(1)
    train_xs = []
    for i in range(train_x.shape[0]):
        a, b = np.random.randint(0, 2), np.random.randint(0, 2)
        x = train_x[i].reshape([28, 28])
        if a==1:
            x = x[::-1]
        if b==1:
            x = x[:, ::-1]
        train_xs.append(x)
    train_x = np.stack(train_xs)
    train_x = np.minimum(np.maximum(train_x, 0.), 1.)

    test_xs = []
    for i in range(test_x.shape[0]):
        a, b = np.random.randint(0, 2), np.random.randint(0, 2)
        x = test_x[i].reshape([28, 28])
        if a==1:
            x = x[::-1]
        if b==1:
            x = x[:, ::-1]
        test_xs.append(x)

    test_x = np.stack(test_xs)
    test_x = np.minimum(np.maximum(test_x, 0.), 1.)

    perm = np.random.permutation(train_x.shape[0])
    save_image_collections(train_x[perm[:100]][..., None],
                           osp.join(root_path, 'data', 'flipmnist_train.png'), shape=[10, 10])

    perm = np.random.permutation(test_x.shape[0])
    save_image_collections(test_x[perm[:100]][..., None],
                           osp.join(root_path, 'data', 'flipmnist_test.png'), shape=[10, 10])

    train_x = train_x.reshape([train_x.shape[0], -1])
    test_x = test_x.reshape([test_x.shape[0], -1])
    with open(osp.join(root_path, 'data', 'flipmnist.npz'), 'wb') as file:
        np.savez(file, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def save_translate_mnist():
    train_x, train_y, test_x, test_y = load_mnist()

    np.random.seed(1)
    train_xs = []
    for i in range(train_x.shape[0]):
        a, b = np.random.randint(0, 28), np.random.randint(0, 28)
        x = train_x[i].reshape([28, 28])
        x = np.roll(x, a, axis=0)
        x = np.roll(x, b, axis=1)
        train_xs.append(x)
    train_x = np.stack(train_xs)

    test_xs = []
    for i in range(test_x.shape[0]):
        a, b = np.random.randint(0, 28), np.random.randint(0, 28)
        x = test_x[i].reshape([28, 28])
        x = np.roll(x, a, axis=0)
        x = np.roll(x, b, axis=1)
        test_xs.append(x)

    test_x = np.stack(test_xs)

    perm = np.random.permutation(train_x.shape[0])
    save_image_collections(train_x[perm[:100]][..., None],
                           osp.join(root_path, 'data', 'translatemnist_train.png'),
                           shape=[10, 10])

    perm = np.random.permutation(test_x.shape[0])
    save_image_collections(test_x[perm[:100]][..., None],
                           osp.join(root_path, 'data', 'translatemnist_test.png'),
                           shape=[10, 10])

    train_x = train_x.reshape([train_x.shape[0], -1])
    test_x = test_x.reshape([test_x.shape[0], -1])
    with open(osp.join(root_path, 'data', 'translatemnist.npz'), 'wb') as file:
        np.savez(file, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


if __name__ == '__main__':
    save_randomflip_mnist()
    # save_translate_mnist()
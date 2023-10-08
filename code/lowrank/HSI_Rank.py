#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import glob
import numpy as np
import cv2
import scipy.linalg as lin

np.random.seed(2021)

def read_HSI():
    blank = np.zeros([256, 256, 32])
    path = sorted(glob.glob('./complete_ms_data/balloons_ms/balloons_ms/*.png'))

    for i in range(len(path)):
        hsi_c = cv2.imread(path[i], cv2.IMREAD_GRAYSCALE)
        hsi_c = cv2.pyrDown(hsi_c)
        blank[:, :, i:i + 1] = hsi_c[:, :, None]

    return blank


def Blur_HSI():
    blank = np.zeros([256, 256, 32])
    path = sorted(glob.glob('./complete_ms_data/balloons_ms/balloons_ms/*.png'))

    for i in range(len(path)):
        hsi_c = cv2.imread(path[i], cv2.IMREAD_GRAYSCALE)
        hsi_c = cv2.pyrDown(hsi_c)
        if i % 3 == 0:
            hsi_blur = cv2.GaussianBlur(hsi_c, ksize=(7, 7), sigmaX=0, sigmaY=0)
            blank[:, :, i:i + 1] = hsi_blur[:, :, None]


    return blank


def loss_bank(file, rank, loss):
    file.write('rank = ' + str(rank) + ' loss = ' + str(loss) + '\n')


def tensor_visual(tensor):
    for i in range(32):
        hsi_c = tensor[:, :, i: i + 1]
        cv2.imwrite('./restore/' + str(i) + '.png', hsi_c)


def col_normalize(X):
    return X / np.sqrt(np.sum(X ** 2, axis=0))


def tensor2matrix(X, mode):
    """
    X shape: (40, 80, 120)
    Return mu-mode matricization from a given tensor
    """
    num_dim = len(X.shape)
    n = X.shape[num_dim - mode]
    X = np.moveaxis(X, num_dim - mode, -1)
    return np.reshape(X, (-1, n)).T


def matrix2tensor(X1, out_shape):
    """
    Input: 1-mode matricization
    Output: tensor with size like (n3, n2, n1)
    """
    return np.reshape(X1.T, out_shape)


def ALS_solver(X, r, nmax=2000, err_tol=1e-4):
    """
    Parameters
    X : tensor like B1
    r : tensor rank
    nmax : maximum number of iterations
    err_tol : tolerance for relative residual error, optional
    Returns：approximated tensor with same shape as X
    """
    ############################# 第一种 ###############################
    # n1, n2, n3 = X.shape
    # B = np.random.random((n2, r))
    # C = np.random.random((n3, r))
    # X1 = tl.unfold(X, mode=0)  # (4, 96)这里调用tensorly包里面的，结果是一样的。
    # X2 = tl.unfold(X, mode=1)  # (8, 48)这里调用tensorly包里面的，结果是一样的。
    # X3 = tl.unfold(X, mode=2)  # (12, 32)这里调用tensorly包里面的，结果是一样的。

    # B = np.random.normal(0, 1, (n2, r))
    # C = np.random.normal(0, 1, (n3, r)) # 这种和下面都是可以的。
    # X1 = tensor2matrix(X, 3)
    # X2 = tensor2matrix(X, 2)
    # X3 = tensor2matrix(X, 1)
    ############################# 第二种 ###############################
    n3, n2, n1 = X.shape
    B = np.random.normal(0, 1, (n2, r))
    C = np.random.normal(0, 1, (n3, r)) # 这种和下面都是可以的。
    X1 = tensor2matrix(X, 1)
    X2 = tensor2matrix(X, 2)
    X3 = tensor2matrix(X, 3)

    X_norm = lin.norm(X1, 'fro')
    err = np.inf
    B = col_normalize(B)
    i = 0
    while (err >= err_tol) and i < nmax:
        C = col_normalize(C)
        tem1 = lin.khatri_rao(C, B)
        A, res, rnk, s = lin.lstsq(tem1, X1.T)
        A = A.T

        A = col_normalize(A)
        tem2 = lin.khatri_rao(C, A)
        B, res, rnk, s = lin.lstsq(tem2, X2.T)
        B = B.T

        B = col_normalize(B)
        tem3 = lin.khatri_rao(B, A)
        C, res, rnk, s = lin.lstsq(tem3, X3.T)
        C = C.T

        X_hat1 = A.dot(lin.khatri_rao(C, B).T)
        err = lin.norm(X_hat1 - X1, 'fro') / X_norm
        i += 1
        print('Relative error at iteration ', i, ': ', err)

    X_hat = matrix2tensor(X_hat1, X.shape)

    print('Finished!')
    return A, B, C, X_hat, err


if __name__ == "__main__":
    tensor = read_HSI()
    tensor_blur = Blur_HSI()

    # result = open('./result.txt', 'a')
    #
    # result.write('Normal HSI:\n')
    # for i in range(3, 21):
    #     A, B, C, X_hat, loss = ALS_solver(tensor, r=i)
    #     loss_bank(result, i, loss)
    #
    # result.write('\nBlur HSI:\n')
    # for j in range(3, 21):
    #     A, B, C, X_hat, loss = ALS_solver(tensor_blur, r=j)
    #     loss_bank(result, j, loss)
    #
    # result.close()

    A, B, C, X_hat, loss = ALS_solver(tensor, r=8)
    tensor_visual(X_hat)

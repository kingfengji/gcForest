# -*- coding:utf-8 -*-
import numpy as np
from joblib import Parallel, delayed

from .log_utils import get_logger

LOGGER = get_logger('win.win_helper')

def get_windows_channel(X, X_win, des_id, nw, nh, win_x, win_y, stride_x, stride_y):
    """
    X: N x C x H x W
    X_win: N x nc x nh x nw
    (k, di, dj) in range(X.channle, win_y, win_x)
    """
    #des_id = (k * win_y + di) * win_x + dj
    dj = int(des_id % win_x)
    di = int(des_id / win_x % win_y)
    k = int(des_id / win_x / win_y)
    src = X[:, k, di:di+nh*stride_y:stride_y, dj:dj+nw*stride_x:stride_x].ravel()
    des = X_win[des_id, :]
    np.copyto(des, src)

def get_windows(X, win_x, win_y, stride_x=1, stride_y=1, pad_x=0, pad_y=0):
    """
    parallizing get_windows
    Arguments:
        X (ndarray): n x c x h x w
    Return:
        X_win (ndarray): n x nh x nw x nc
    """
    assert len(X.shape) == 4
    n, c, h, w = X.shape
    if pad_y > 0:
        X = np.concatenate(( X, np.zeros((n, c, pad_y, w),dtype=X.dtype) ), axis=2)
        X = np.concatenate(( np.zeros((n, c, pad_y, w),dtype=X.dtype), X ), axis=2)
    n, c, h, w = X.shape
    if pad_x > 0:
        X = np.concatenate(( X, np.zeros((n, c, h, pad_x),dtype=X.dtype) ), axis=3)
        X = np.concatenate(( np.zeros((n, c, h, pad_x),dtype=X.dtype), X ), axis=3)
    n, c, h, w = X.shape
    nc = win_y * win_x * c
    nh = int((h - win_y) / stride_y + 1)
    nw = int((w - win_x) / stride_x + 1)
    X_win = np.empty(( nc, n * nh * nw), dtype=np.float32)
    LOGGER.info("get_windows_start: X.shape={}, X_win.shape={}, nw={}, nh={}, c={}, win_x={}, win_y={}, stride_x={}, stride_y={}".format(
                X.shape, X_win.shape, nw, nh, c, win_x, win_y, stride_x, stride_y))
    Parallel(n_jobs=-1, backend="threading", verbose=0)(
            delayed(get_windows_channel)(X, X_win, des_id, nw, nh, win_x, win_y, stride_x, stride_y)
            for des_id in range(c * win_x * win_y))
    LOGGER.info("get_windows_end")
    X_win = X_win.transpose((1, 0))
    X_win = X_win.reshape((n, nh, nw, nc))
    return X_win

def calc_accuracy(y_gt, y_pred, tag):
    LOGGER.info("Accuracy({})={:.2f}%".format(tag, np.sum(y_gt==y_pred)*100./len(y_gt)))

def win_vote(y_win_predict, n_classes):
    """ 
     
    y_win_predict (ndarray): n x n_window
        y_win_predict[i, j] prediction for the ith data of jth window 
    """
    y_pred = np.zeros(len(y_win_predict), dtype=np.int16)
    for i, y_bag in enumerate(y_win_predict):
        y_pred[i] = np.argmax(np.bincount(y_bag,minlength=n_classes))
    return y_pred

def win_avg(y_win_proba):
    """ 
     
    Parameters
    ----------
    y_win_proba: n x n_windows x n_classes
    """
    n_classes = y_win_proba.shape[-1]
    y_bag_proba = np.mean(y_win_proba, axis=1)
    y_pred = np.argmax(y_bag_proba, axis=1)
    return y_pred

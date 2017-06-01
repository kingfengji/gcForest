# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import numpy as np
import os.path as osp
from keras.datasets import mnist

from .ds_base import ds_base

class MNIST(ds_base):
    def __init__(self, **kwargs):
        super(MNIST, self).__init__(**kwargs)
        # data_path = osp.abspath( osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir,
        #     'datasets/mnist/keras/mnist.pkl.gz') )
        # with gzip.open(data_path, 'rb') as f:
        #     (X_train, y_train), (X_test, y_test) = pickle.load(f)
        #
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        if self.data_set == 'train':
            X = X_train
            y = y_train
        elif self.data_set == 'train-small':
            X = X_train[:2000]
            y = y_train[:2000]
        elif self.data_set == 'test':
            X = X_test
            y = y_test
        elif self.data_set == 'test-small':
            X = X_test[:1000]
            y = y_test[:1000]
        elif self.data_set == 'all':
            X = np.vstack((X_train, X_test))
            y = np.vstack((y_train, y_test))
        else:
            raise ValueError('MNIST Unsupported data_set: ', self.data_set)

        # normalization
        if self.norm:
            X = X.astype(np.float32) / 255
        X = X[:,np.newaxis,:,:]
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y


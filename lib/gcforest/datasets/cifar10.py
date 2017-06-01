# encoding: utf-8
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""

import numpy as np
from .ds_base import ds_base
from keras.datasets import cifar10

cls_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""
X_train.shape: (50000, 3, 32, 32)
X_test.shape: (10000, 3, 32, 32)
y: 10 labels
"""
class CIFAR10(ds_base):
    def __init__(self, **kwargs):
        super(CIFAR10, self).__init__(**kwargs)
        self.cls_names = cls_names
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape((y_train.shape[0]))
        y_test = y_test.reshape((y_test.shape[0]))
        if self.data_set == 'train':
            X = X_train
            y = y_train
        elif self.data_set == 'train-small':
            X = X_train[:1000]
            y = y_train[:1000]
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
        if X.shape[-1] == 3:
            X = X.transpose((0, 3, 1, 2))
        # normalization
        if self.norm:
            X = X.astype(np.float32) / 255
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y


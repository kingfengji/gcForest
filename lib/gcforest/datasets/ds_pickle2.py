"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
from __future__ import print_function
import pickle
import os, os.path as osp

class DSPickle2(object):
    def __init__(self, data_path, X_keys):
        self.data_path = data_path
        print('Loading data from {}'.format(data_path))
        with open(data_path) as f:
            datas = pickle.load(f)
        self.X = []
        for X_key in X_keys:
            self.X.append(datas[X_key])
        self.y = datas["y"]
        print('Data Loaded (X.shape={}, y.shape={})'.format([x1.shape for x1 in self.X], self.y.shape))

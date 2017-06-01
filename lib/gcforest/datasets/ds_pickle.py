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
from .ds_base import ds_base
"""
Using cPickle to save and load dataset
"""

def save_dataset(data_path, X, y):
    print('Data Saving in {} (X.shape={},y.shape={})'.format(
            data_path, X.shape, y.shape))
    data_dir = osp.abspath(osp.join(data_path, osp.pardir))
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    data = {'X': X, 'y': y}
    with open(data_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_dataset(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    print('Data Loaded from {} (X.shape={}, y.shape={})'.format(data_path, X.shape, y.shape))
    return X, y

class DSPickle(ds_base):
    def __init__(self, data_path, **kwargs):
        super(DSPickle, self).__init__(**kwargs)
        self.data_path = data_path
        X, y = load_dataset(data_path)

        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y

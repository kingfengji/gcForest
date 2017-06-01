# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""

import os.path as osp
import numpy as np

class ds_base(object):
    def __init__(self, data_set="train", norm=False, layout_x="tensor", layout_y="label", conf=None):
        self.conf = conf
        if conf is not None:
            self.data_set = conf["data_set"]
            self.norm = int(conf.get("norm", 0))
            self.layout_x = conf.get("layout_x", "tensor")
            self.layout_y = conf.get("layout_y", "label")
        else:
            self.data_set = data_set
            self.norm = norm
            self.layout_x = layout_x
            self.layout_y = layout_y

    @property
    def n_classes(self):
        if hasattr(self, "n_classes_"):
            return self.n_classes_
        return len(np.unique(self.y))

    def init_layout_X(self, X):
        """
        input X format: tensor
        """
        # reshape X
        if self.layout_x == "tensor":
            pass
        elif self.layout_x == "vector":
            X = X.reshape((X.shape[0], -1))
        elif self.layout_x == "sequence":
            assert X.shape[3] == 1
            X = X[:,:,:,0].transpose((0,2,1))
        else:
            raise ValueError("DataSet doesn't supported layout_x: ", self.layout_x)
        return X

    def init_layout_y(self, y, X=None):
        # reshape y
        if self.layout_y == "label":
            pass
        elif self.layout_y == "bin":
            from keras.utils import np_utils
            y = np_utils.to_categorical(y)
        elif self.layout_y == "autoencoder":
            y = X
        else:
            raise ValueError("MNIST Unsupported layout_y: ", self.layout_y)
        return y

    def get_data_by_imageset(self, X_train, y_train, X_test, y_test):
        if self.data_set == "train":
            X = X_train
            y = y_train
        elif self.data_set == "test":
            X = X_test
            y = y_test
        elif self.data_set == "all":
            X = np.vstack((X_train, X_test))
            y = np.vstack((y_train, y_test))
        else:
            raise ValueError("Unsupported data_set: ", self.data_set)
        return X, y

def get_dataset_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir, "datasets"))

def get_dataset_cache_base():
    return osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir, osp.pardir, "datasets-cache"))

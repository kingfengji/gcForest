"""
Description: A python 2.7 implementation of gcForest proposed in [1]. 
A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. 
The implementation is flexible enough for modifying the model or fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

from .ds_base import ds_base

def load_data(train_num, train_repeat):
    test_size = (10. - train_num) / 10
    data = fetch_olivetti_faces()
    X = data.images
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=3, stratify=y)
    if train_repeat > 1:
        X_train = X_train.repeat(train_repeat, axis=0)
        y_train = y_train.repeat(train_repeat)
    return X_train, y_train, X_test, y_test

class OlivettiFace(ds_base):
    def __init__(self, train_num=5, train_repeat=1, **kwargs):
        """
        train_num: int
        """
        super(OlivettiFace, self).__init__(**kwargs)

        X_train, y_train, X_test, y_test = load_data(train_num, train_repeat)
        X, y = self.get_data_by_imageset(X_train, y_train, X_test, y_test)

        X = X[:,np.newaxis,:,:]
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y

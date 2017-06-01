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

from .ds_base import ds_base
from .ds_base import get_dataset_base


def load_data():
    data_path = osp.join(get_dataset_base(), "uci_letter", "letter-recognition.data")
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines()]
    n_datas = len(rows)
    X = np.zeros((n_datas, 16), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        X[i, :] = list(map(float, row[1:]))
        y[i] = ord(row[0]) - ord('A')
    X_train, y_train = X[:16000], y[:16000]
    X_test, y_test = X[16000:], y[16000:]
    return X_train, y_train, X_test, y_test

class UCILetter(ds_base):
    def __init__(self, **kwargs):
        super(UCILetter, self).__init__(**kwargs)
        X_train, y_train, X_test, y_test = load_data()
        X, y = self.get_data_by_imageset(X_train, y_train, X_test, y_test)

        X = X[:,np.newaxis,:,np.newaxis]
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y

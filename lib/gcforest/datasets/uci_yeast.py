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
import os.path as osp
import re
from sklearn.model_selection import train_test_split

from .ds_base import ds_base, get_dataset_base


def load_data():
    id2label = {}
    label2id = {}
    label_path = osp.abspath( osp.join(get_dataset_base(), "uci_yeast", "yeast.label") )
    with open(label_path) as f:
        for row in f:
            cols = row.strip().split(" ")
            id2label[int(cols[0])] = cols[1]
            label2id[cols[1]] = int(cols[0])

    data_path = osp.abspath( osp.join(get_dataset_base(), "uci_yeast", "yeast.data") )
    with open(data_path) as f:
        rows = f.readlines()
    n_datas = len(rows)
    X = np.zeros((n_datas, 8), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        cols = re.split(" +", row.strip())
        #print(list(map(float, cols[1:1+8])))
        X[i,:] = list(map(float, cols[1:1+8]))
        y[i] = label2id[cols[-1]]
    train_idx, test_idx = train_test_split(range(n_datas), random_state=0, train_size=0.7, stratify=y)
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


class UCIYeast(ds_base):
    def __init__(self, **kwargs):
        super(UCIYeast, self).__init__(**kwargs)
        (X_train, y_train), (X_test, y_test) = load_data()
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
            raise ValueError("YEAST Unsupported data_set: ", self.data_set)

        X = X[:,np.newaxis,:,np.newaxis]
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y


# encoding: utf-8
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

class FeatureParser(object):
    def __init__(self, desc):
        desc = desc.strip()
        if desc == "C":
            self.f_type = "number"
        else:
            self.f_type = "categorical"
            f_names = [d.strip() for d in desc.split(",")]
            # missing value
            f_names.insert(0, "?")
            self.name2id = dict(zip(f_names, range(len(f_names))))

    def get_float(self, f_data):
        f_data = f_data.strip()
        if self.f_type == "number":
            return float(f_data)
        return float(self.name2id[f_data])

    def get_data(self, f_data):
        f_data = f_data.strip()
        if self.f_type == "number":
            return float(f_data)
        data = np.zeros(len(self.name2id), dtype=np.float32)
        data[self.name2id[f_data]] = 1
        return data

    def get_fdim(self):
        """
        get feature dimension
        """
        if self.f_type == "number":
            return 1
        return len(self.name2id)


def load_data(data_set, cate_as_onehot):
    if data_set == "train":
        data_path = osp.join(get_dataset_base(), "uci_adult", "adult.data")
    elif data_set == "test":
        data_path = osp.join(get_dataset_base(), "uci_adult", "adult.test")
    else:
        raise ValueError("Unkown data_set: ", data_set)
    f_parsers = []
    feature_desc_path = osp.join(get_dataset_base(), "uci_adult", "features")
    with open(feature_desc_path) as f:
        for row in f.readlines():
            f_parsers.append(FeatureParser(row))

    with open(data_path) as f:
        rows = [row.strip().split(",") for row in f.readlines() if len(row.strip()) > 0 and not row.startswith("|")]
    n_datas = len(rows)
    if cate_as_onehot:
        X_dim = np.sum([f_parser.get_fdim() for f_parser in f_parsers])
        X = np.zeros((n_datas, X_dim), dtype=np.float32)
    else:
        X = np.zeros((n_datas, 14), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        assert len(row) == 15, "len(row) wrong, i={}".format(i)
        foffset = 0
        for j in range(14):
            if cate_as_onehot:
                fdim = f_parsers[j].get_fdim()
                X[i, foffset:foffset+fdim] = f_parsers[j].get_data(row[j].strip())
                foffset += fdim
            else:
                X[i, j] = f_parsers[j].get_float(row[j].strip())
        y[i] = 0 if row[-1].strip().startswith("<=50K") else 1
    return X, y


class UCIAdult(ds_base):
    def __init__(self, cate_as_onehot=0, **kwargs):
        """
        cate_as_number: 是否把包含离散取值的feature用onehot表示
        """
        super(UCIAdult, self).__init__(**kwargs)
        if self.conf is not None:
            cate_as_onehot = int(self.conf.get("cate_as_onehot", 0))
        X, y = load_data(self.data_set, cate_as_onehot)

        X = X[:,np.newaxis,:,np.newaxis]
        X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y

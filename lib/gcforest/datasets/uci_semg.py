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
import scipy.io as sio
from sklearn.model_selection import train_test_split

from .ds_base import ds_base
from .ds_base import get_dataset_base

move2label = {}
move2label['spher_ch1'] = 0
move2label['spher_ch2'] = 0
move2label['tip_ch1'] = 1
move2label['tip_ch2'] = 1
move2label['palm_ch1'] = 2
move2label['palm_ch2'] = 2
move2label['lat_ch1'] = 3
move2label['lat_ch2'] = 3
move2label['cyl_ch1'] = 4
move2label['cyl_ch2'] = 4
move2label['hook_ch1'] = 5
move2label['hook_ch2'] = 5

def load_mat(mat_path):
    X = None
    y = None
    data = sio.loadmat(mat_path)
    for k in sorted(move2label.keys()):
        X_cur = data[k]
        y_cur = np.full(X_cur.shape[0], move2label[k], dtype=np.int32)
        if X is None:
            X, y = X_cur, y_cur
        else:
            X = np.vstack((X, X_cur))
            y = np.concatenate((y, y_cur))
    return X, y

def load_data():
    db_base = osp.join(get_dataset_base(), 'uci_semg', 'Database 1')
    X = None
    y = None
    for mat_name in ('female_1.mat', 'female_2.mat', 'female_3.mat', 'male_1.mat', 'male_2.mat'):
        X_cur, y_cur = load_mat(osp.join(db_base, mat_name))
        if X is None:
            X, y = X_cur, y_cur
        else:
            X = np.vstack((X, X_cur))
            y = np.concatenate((y, y_cur))
    n_datas = X.shape[0]
    train_idx, test_idx = train_test_split(range(n_datas), random_state=0,
            train_size=0.7, stratify=y)
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])

class UCISEMG(ds_base):
    def __init__(self, **kwargs):
        super(UCISEMG, self).__init__(**kwargs)
        (X_train, y_train), (X_test, y_test) = load_data()
        X, y = self.get_data_by_imageset(X_train, y_train, X_test, y_test)

        X = X[:,np.newaxis,:,np.newaxis]
        if self.layout_x == 'lstm':
            X = X.reshape((X.shape[0], -1, 6)).transpose((0, 2, 1))
        else:
            X = self.init_layout_X(X)
        y = self.init_layout_y(y)
        self.X = X
        self.y = y

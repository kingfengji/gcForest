"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import numpy as np
import os, os.path as osp
import sys
from .ds_base import ds_base, get_dataset_base, get_dataset_cache_base

DEFAULT_DATA_BASE = osp.abspath( osp.join(get_dataset_base(),'gtzan','genres') )
DEFAULT_IMAGEST_BASE = osp.abspath( osp.join(get_dataset_base(),'gtzan','splits') )
DEFAULT_CACHE_BASE = osp.abspath( osp.join(get_dataset_cache_base(),'gtzan') )
DEFAULT_GENRE_LIST = (
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock',
        )

def parse_anno_file(anno_path):
    X = []
    y = []
    with open(anno_path, 'r') as f:
        for row in f:
            cols = row.strip().split(' ')
            X.append(cols[0])
            y.append(int(cols[1]))
    y = np.asarray(y, dtype=np.int16)
    return X, y

def read_data(anno_path, mode, genre_base=None):
    genre_base = genre_base or DEFAULT_DATA_BASE
    au_path_list = []
    y = []
    with open(anno_path) as f:
        for row in f:
            cols = row.strip().split(' ')
            au_path = osp.join(genre_base, cols[0])
            au_path_list.append(au_path)
            y.append(int(cols[1]))
    if mode == 'fft':
        X = Parallel(n_jobs=-1, backend='threading')(
                delayed(get_fft_feature)(au_path, 1000)
                for i, au_path in enumerate(au_path_list))
    elif mode == 'ceps':
        X = Parallel(n_jobs=-1, backend='threading')(
                delayed(get_ceps_feature)(au_path)
                for i, au_path in enumerate(au_path_list))
    else:
        raise ValueError('Unkown mode: ', mode)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

class GTZAN(ds_base):
    def __init__(self, cache=None, **kwargs):
        super(GTZAN, self).__init__(**kwargs)
        if kwargs.get('conf') is not None:
            conf = kwargs['conf']
            cache = conf.get('cache', None)
        data_set_path = osp.join(DEFAULT_IMAGEST_BASE, self.data_set)
        self.data_set_path = data_set_path
        self.cache = cache
        X, y = parse_anno_file(data_set_path)
        if cache == 'raw':
            import librosa
            from tqdm import trange
            X_new = np.zeros((len(X), 1, 661500, 1))
            for i in trange(len(X)):
                x,_ = librosa.load(osp.join(DEFAULT_DATA_BASE, X[i]))
                x_len = min(661500, len(x))
                X_new[i,:,:x_len,0] = x[:x_len]
        if cache is not None and cache != 'raw':
            X = self.load_cache_X(X, cache)
            if cache == 'mfcc':
                X_new = np.zeros((len(X), X[0].shape[0], 1280, 1))
                for i, x in enumerate(X):
                    x_len = min(x.shape[1], 1280)
                    X_new[i,:,:x_len,0] = x[:,:x_len]
                X = X_new

        # layout_X
        if self.layout_x == 'rel_path':
            self.X = X
        else:
            self.X = self.init_layout_X(X)
        # layout_y
        self.y = self.init_layout_y(y)

    def load_cache_X(self, rel_paths, cache_name):
        X = []
        for rel_path in rel_paths:
            cache_path = osp.join(self.cache_base, cache_name, osp.splitext(rel_path)[0] + '.npy')
            X.append(np.load(cache_path))
        return X

    @property
    def cache_base(self):
        return DEFAULT_CACHE_BASE
    
    @property
    def data_base(self):
        return DEFAULT_DATA_BASE

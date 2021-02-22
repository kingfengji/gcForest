# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import numpy as np
#from tqdm import trange

from .base_layer import BaseLayer
from ..utils.debug_utils import repr_blobs_shape
from ..utils.log_utils import get_logger

LOGGER = get_logger('gcforest.layers.fg_pool_layer')

class FGPoolLayer(BaseLayer):
    def __init__(self, layer_config, data_cache):
        """
        Pooling Layer (MaxPooling, AveragePooling)
        """
        super(FGPoolLayer, self).__init__(layer_config, data_cache)
        self.win_x = self.get_value("win_x", None, int, required=True)
        self.win_y = self.get_value("win_y", None, int, required=True)
        self.pool_method = self.get_value("pool_method", "avg", str)

    def fit_transform(self, train_config):
        LOGGER.info("[data][{}] bottoms={}, tops={}".format(self.name, self.bottom_names, self.top_names))
        self._transform(train_config.phases, True)

    def transform(self):
        LOGGER.info("[data][{}] bottoms={}, tops={}".format(self.name, self.bottom_names, self.top_names))
        self._transform(["test"], False)

    def _transform(self, phases, check_top_cache):
        for ti, top_name in enumerate(self.top_names):
            LOGGER.info("[progress][{}] ti={}/{}, top_name={}".format(ti, self.name, len(self.top_names), top_name))
            for phase in phases:
                # check top cache
                if check_top_cache and self.check_top_cache([phase], ti)[0]:
                    continue
                X = self.data_cache.get(phase, self.bottom_names[ti])
                LOGGER.info('[data][{},{}] bottoms[{}].shape={}'.format(self.name, phase, ti, X.shape))
                n, c, h, w = X.shape
                win_x, win_y = self.win_x, self.win_y
                #assert h % win_y == 0
                #assert w % win_x == 0
                #nh = int(h / win_y)
                #nw = int(w / win_x)
                nh = int((h - 1) / win_y + 1)
                nw = int((w - 1) / win_x + 1)
                X_pool = np.empty(( n, c, nh, nw), dtype=np.float32)
                #for k in trange(c, desc='loop channel'):
                #    for di in trange(nh, desc='loop win_y'):
                #        for dj in trange(nw, desc='loop win_x'):
                for k in range(c):
                    for di in range(nh):
                        for dj in range(nw):
                            si = di * win_y
                            sj = dj * win_x
                            src = X[:, k, si:si+win_y, sj:sj+win_x]
                            src = src.reshape((X.shape[0], -1))
                            if self.pool_method == 'max':
                                X_pool[:, k, di, dj] = np.max(src, axis=1)
                            elif self.pool_method == 'avg':
                                X_pool[:, k, di, dj] = np.mean(src, axis=1)
                            else:
                                raise ValueError('Unkown Pool Method, pool_method={}'.format(self.pool_method))
                #print ('\n')
                LOGGER.info('[data][{},{}] tops[{}].shape={}'.format(self.name, phase, ti, X_pool.shape))
                self.data_cache.update(phase, top_name, X_pool)

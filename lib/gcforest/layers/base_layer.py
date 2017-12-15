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

from ..utils.log_utils import get_logger
from ..utils.config_utils import get_config_value

LOGGER = get_logger('gcforest.layers.base_layer')

class BaseLayer(object):
    def __init__(self, layer_config, data_cache):
        self.layer_config = layer_config
        self.name = layer_config["name"]
        self.bottom_names = layer_config["bottoms"]
        self.top_names = layer_config["tops"]
        self.data_cache = data_cache

    def get_value(self, key, default_value, value_types, required=False, config=None):
        return get_config_value(config or self.layer_config, key, default_value, value_types, 
                required=required, config_name=self.name)
        return value

    def check_top_cache(self, phases, ti):
        """
        Check if top cache exists

        Parameters
        ---------
        phases: List of str
            e.g. ["train", "test"]
        ti: int
            top index

        Return
        ------
        exist_mask: List of bool
            exist_mask[ti] represent tops[ti] is exist in cache (either keeped in memory or saved in disk)
        """
        top_name = self.top_names[ti]
        exist_mask = np.zeros(len(phases))
        for pi, phase in enumerate(phases):
            top = self.data_cache.get(phase, top_name, ignore_no_exist=True)
            exist_mask[pi] = top is not None
            if top is not None:
                LOGGER.info("[data][{},{}] top cache exists. tops[{}].shape={}".format(self.name, phase, ti, top.shape))
        return exist_mask

    def fit_transform(self, train_config):
        raise NotImplementedError()

    def transform(self):
        raise NotImplementedError()

    def score(self):
        pass

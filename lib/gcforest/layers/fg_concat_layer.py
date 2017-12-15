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

from .base_layer import BaseLayer
from ..utils.debug_utils import repr_blobs_shape
from ..utils.log_utils import get_logger

LOGGER = get_logger('gcforest.layers.fg_concat_layer')

class FGConcatLayer(BaseLayer):
    def __init__(self, layer_config, data_cache):
        """
        Concat Layer
        """
        super(FGConcatLayer, self).__init__(layer_config, data_cache)
        self.axis = self.get_value("axis", -1, int)
        assert(len(self.bottom_names) > 0)
        assert(len(self.top_names) == 1)

    def fit_transform(self, train_config):
        LOGGER.info("[data][{}] bottoms={}, tops={}".format(self.name, self.bottom_names, self.top_names))
        self._transform(train_config.phases)

    def transform(self):
        LOGGER.info("[data][{}] bottoms={}, tops={}".format(self.name, self.bottom_names, self.top_names))
        self._transform(["test"])

    def _transform(self, phases):
        """
        bottoms:
            for example: n x Ci x w x h
        """
        for phase in phases:
            # check top cache
            if self.check_top_cache([phase], 0)[0]:
                continue
            bottoms = self.data_cache.gets(phase, self.bottom_names)
            LOGGER.info('[data][{},{}] bottoms.shape={}'.format(self.name, phase, repr_blobs_shape(bottoms)))
            if self.axis == -1:
                for i, bottom in enumerate(bottoms):
                    bottoms[i] = bottom.reshape((bottom.shape[0], -1))
                concat_data = np.concatenate(bottoms, axis=1)
            else:
                concat_data = np.concatenate(bottoms, axis=self.axis)
            LOGGER.info('[data][{},{}] tops[0].shape={}'.format(self.name, phase, concat_data.shape))
            self.data_cache.update(phase, self.top_names[0], concat_data)

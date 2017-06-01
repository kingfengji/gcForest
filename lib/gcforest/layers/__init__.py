# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
from .base_layer import BaseLayer
from .fg_concat_layer import FGConcatLayer
from .fg_pool_layer import FGPoolLayer
from .fg_win_layer import FGWinLayer

def get_layer_class(layer_type):
    if layer_type == "FGWinLayer":
        return FGWinLayer
    if layer_type == "FGConcatLayer":
        return FGConcatLayer
    if layer_type == "FGPoolLayer":
        return FGPoolLayer
    raise ValueError("Unkown Layer Type: ", layer_type)

def get_layer(layer_config, data_cache):
    """
    layer_config (dict): config for layer 
    data_cache (gcforest.DataCache): DataCache 
    """
    layer_config = layer_config.copy()
    layer_class = get_layer_class(layer_config["type"])
    layer_config.pop("type")
    layer = layer_class(layer_config, data_cache)
    return layer

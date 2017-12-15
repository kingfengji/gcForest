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
import os, os.path as osp
import json

from .layers import get_layer
from .utils.log_utils import get_logger

LOGGER = get_logger("gcforest.gcnet")

class FGNet(object):
    """
    GCForest : FineGrained Components
    """
    def __init__(self, net_config, data_cache):
        #net_config_str = json.dumps(net_config, sort_keys=True, indent=4, separators=(',', ':'))
        #LOGGER.info("\n" + net_config_str)
        self.data_cache = data_cache
        self.inputs = net_config.get("inputs", [])
        self.check_net_config(net_config)
        self.outputs = net_config.get("outputs", [])

        # layers
        self.layers = []
        self.name2layer = {}
        model_disk_base = net_config.get("model_cache", {}).get("disk_base", None)
        for layer_config in net_config["layers"]:
            layer = get_layer(layer_config, self.data_cache)
            layer.model_disk_base = model_disk_base
            self.layers.append(layer)
            self.name2layer[layer.name] = layer


    def fit_transform(self, X_train, y_train, X_test, y_test, train_config):
        """
        delete_layer (bool): defalut=False
            When X_test is not None and there is no need to run test, delete layer in realtime to save mem
             
        """
        LOGGER.info("X_train.shape={}, y_train.shape={}, X_test.shape={}, y_test.shape={}".format(
            X_train.shape, y_train.shape, None if X_test is None else X_test.shape, None if y_test is None else y_test.shape))
        self.update_xy("train", X_train, y_train)
        if "test" in train_config.phases:
            self.update_xy("test", X_test, y_test)
        for li, layer in enumerate(self.layers):
            layer.fit_transform(train_config)

    @staticmethod
    def concat_datas(datas):
        if type(datas) != list:
            return datas
        for i, data in enumerate(datas):
            datas[i] = data.reshape((data.shape[0], -1))
        return np.concatenate(datas, axis=1)

    def transform(self, X_test):
        LOGGER.info("X_test.shape={}".format(X_test.shape))
        self.data_cache.update("test", "X", X_test)
        for li, layer in enumerate(self.layers):
            layer.transform()
        return self.get_outputs("test")

    def score(self):
        for li, layer in enumerate(self.layers):
            layer.score()

    def update_xy(self, phase, X, y):
        self.data_cache.update(phase, "X", X)
        self.data_cache.update(phase, "y", y)

    def get_outputs(self, phase):
        outputs = self.data_cache.gets(phase, self.outputs)
        return outputs

    def save_outputs(self, phase, save_y=True, save_path=None):
        if save_path is None:
            if self.data_cache.cache_dir is None:
                LOGGER.error("save path is None and data_cache.cache_dir is None!!! don't know where to save")
                return
            save_path = osp.join(self.data_cache.cache_dir, phase, "outputs.pkl")
        import pickle
        info  = ""
        data_names = [name for name in self.outputs]
        if save_y:
            data_names.append("y")
        datas = {}
        for di, data_name in enumerate(data_names):
            datas[data_name] = self.data_cache.get(phase, data_name)
            info = "{},{}->{}".format(info, data_name, datas[data_name].shape)
        LOGGER.info("outputs.shape={}".format(info))
        LOGGER.info("Saving Outputs in {} ".format(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)

    def check_net_config(self, net_config):
        """
        check net_config 
        """
         
        top2layer = {}
        name2layer = {}
        for li, layer_config in enumerate(net_config["layers"]):
            layer_name = layer_config["name"]
            if layer_name in name2layer:
                raise ValueError("layer name duplicate. layer_name={}, config1={}, config2={}".format(
                    layer_name, name2layer[layer_name], layer_config))
            name2layer[layer_name] = layer_config

            for bottom in layer_config["bottoms"]:
                if bottom != "X" and bottom != "y" and not bottom in self.inputs and not bottom in top2layer:
                    raise ValueError("li={}, layer_config={}, bottom({}) doesn't be produced by other layers".format(
                        li, layer_config, bottom))
            for top in layer_config["tops"]:
                if top in top2layer:
                    raise ValueError("top duplicate. layer({}) and layer({}) have same top blob: {}".format(
                        top2layer[top], layer_config["name"], top))
                top2layer[top] = layer_config["name"]
         
        outputs = net_config.get("outputs", [])
        if len(outputs) == 0:
            LOGGER.warn("outputs list is empty!!!")
        for output in outputs:
            if output == "X" or output == "y" or output in self.inputs or output in top2layer:
                continue
            raise ValueError("output data name not exist: output={}".format(output))
         
        for layer_config in net_config["layers"]:
            if len(layer_config["tops"]) > 1:
                for top_name in layer_config["tops"]:
                    if not top_name.startswith(layer_config["name"]):
                        LOGGER.warn("top_name is suggested to startswith layer_name: layer_config={}".format(layer_config))
            else:
                top = layer_config["tops"][0]
                if top != layer_config["name"]:
                    LOGGER.warn("layer_name != top_name, You should check to make sure this is what you want!!! layer_config={}".format(layer_config))

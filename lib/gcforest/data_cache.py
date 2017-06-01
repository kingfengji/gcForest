# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import os, os.path as osp
import numpy as np

from .utils.log_utils import get_logger
from .utils.cache_utils import name2path

LOGGER = get_logger("gcforest.data_cache")

def check_dir(path):
    """ make sure the dir specified by path got created """
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)

def data_disk_path(cache_dir, phase, data_name):
    data_path = osp.join(cache_dir, phase, name2path(data_name) + ".npy")
    return data_path

class DataCache(object):
    def __init__(self, config):
        self.config = config
        self.cache_dir = config.get("cache_dir", None)
        if self.config.get("keep_in_mem") is None:
            self.config["keep_in_mem"] = {"default": 1}
        if self.config.get("cache_in_disk") is None:
            self.config["cache_in_disk"] = {"default": 0}
        self.datas = {"train": {}, "test": {}}

    def keep_in_mem(self, phase, data_name):
        """
        determine if the data for (phase, data_name) should be kept in RAM
        if config["keep_in_mem"][data_name] exist, then use it, otherwise use the default value of config["keep_in_mem"] 
        """
        return self.config["keep_in_mem"].get(data_name, self.config["keep_in_mem"]["default"])

    def cache_in_disk(self, phase, data_name):
        """
        check data for (phase, data_name) is cached in disk
        if config["cache_in_disk"][data_name] exist, then use it , otherwise use default value of config["cache_in_disk"]  
        """
        return self.config["cache_in_disk"].get(data_name, self.config["cache_in_disk"]["default"])

    def is_exist(self, phase, data_name):
        """
        check data_name is generated or cashed to disk 
        """
        data_mem = self.datas[phase].get(data_name, None)
        if data_mem is not None:
            return True
        if self.cache_dir is None:
            return False
        data_path = data_disk_path(self.cache_dir, phase, data_name)
        if osp.exists(data_path):
            return data_path
        return None

    def gets(self, phase, data_names, ignore_no_exist=False):
        assert isinstance(data_names, list)
        datas = []
        for data_name in data_names:
            datas.append(self.get(phase, data_name, ignore_no_exist=ignore_no_exist))
        return datas

    def get(self, phase, data_name, ignore_no_exist=False):
        """
        get data according to data_name 

        Arguments
        ---------
        phase (str): train or test
        data_name (str): name for tops/bottoms  
        ignore_no_exist (bool): if True, when no data found, return None, otherwise raise e
        """
        assert isinstance(data_name, basestring), "data_name={}, type(data_name)={}".format(data_name, type(data_name))
        # return data if data in memory
        data_mem = self.datas[phase].get(data_name, None)
        if data_mem is not None:
            return data_mem
        # load data from disk
        if self.cache_dir is None:
            if ignore_no_exist:
                return None
            raise ValueError("Cache base unset, can't load data ({}->{}) from disk".format(phase, data_name))
        data_path = data_disk_path(self.cache_dir, phase, data_name)
        if not osp.exists(data_path):
            if ignore_no_exist:
                return None
            raise ValueError("Data path not exist, can't load data ({}->{}) from disk: {}".format(phase, data_name, data_path))
        return np.load(data_path)

    def updates(self, phase, data_names, datas):
        assert isinstance(data_names, list)
        for i, data_name in enumerate(data_names):
            self.update(phase, data_name, datas[i])

    def update(self, phase, data_name, data):
        """
        update (phase, data_name) data in cache  
        """
        assert isinstance(data, np.ndarray), "data(type={}) is not a np.ndarray!!!".format(type(data))
        if self.keep_in_mem(phase, data_name):
            self.datas[phase][data_name] = data
        if self.cache_in_disk(phase, data_name):
            if self.cache_dir is None:
                raise ValueError("Cache base unset, can't Save data ({}->{}) to disk".format(phase, data_name))
            data_path = data_disk_path(self.cache_dir, phase, data_name)
            LOGGER.info("Updating data ({}->{}, shape={}) in disk: {}".format(phase, data_name, data.shape, data_path))
            check_dir(data_path);
            np.save(data_path, data)

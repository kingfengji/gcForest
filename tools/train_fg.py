"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import sys, os, os.path as osp
import argparse
import logging
import numpy as np
import json

sys.path.insert(0, 'lib')
from gcforest.utils.log_utils import get_logger, update_default_level, update_default_logging_dir
from gcforest.utils.config_utils import load_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, required=True, help='gcfoest Net Model File')
    parser.add_argument('--save_outputs', dest='save_outputs', action="store_true", help="Save outputs")
    parser.add_argument('--log_dir', dest='log_dir', type=str, default=None, help='Log file directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = load_json(args.model)
    update_default_level(logging.DEBUG)
    if args.log_dir is not None:
        update_default_logging_dir(args.log_dir)
    from gcforest.fgnet import FGNet, FGTrainConfig
    from gcforest.exp_utils import prec_ets, prec_rf, prec_log, prec_xgb, concat_datas
    from gcforest.datasets import get_dataset
    LOGGER = get_logger("tools.train_fg")
    LOGGER.info("tools.train_fg")
    LOGGER.info("\n" + json.dumps(config, sort_keys=True, indent=4, separators=(',', ':')))

    train_config = FGTrainConfig(config["train"])
    if args.save_outputs:
        assert train_config.data_cache.cache_dir is not None, \
                "Data cache dir must be set in model's json config when save_outputs option is on!!"

    data_train = get_dataset(config["dataset"]["train"])
    data_test = get_dataset(config["dataset"]["test"])

    net = FGNet(config["net"], train_config.data_cache)
    net.fit_transform(data_train.X, data_train.y, data_test.X, data_test.y, train_config)

    if args.save_outputs:
        net.save_outputs("train")
        net.save_outputs("test")
    #prec_ets(1000, X_train_enc, data_train.y, X_test_enc, data_test.y, random_state=0)
    #prec_ets(2000, X_train_enc, data_train.y, X_test_enc, data_test.y, random_state=0)
    #prec_xgb(1000, 5, 
    #        concat_datas(net.get_outputs("train")), data_train.y, 
    #        concat_datas(net.get_outputs("test")), data_test.y)
    
    import IPython; IPython.embed()

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
import numpy as np
import xgboost as xgb
sys.path.insert(0, 'lib')

from gcforest.utils.log_utils import get_logger, update_default_level, update_default_logging_dir
from gcforest.fgnet import FGNet, FGTrainConfig
from gcforest.utils.config_utils import load_json
from gcforest.exp_utils import concat_datas
from gcforest.datasets import get_dataset
LOGGER = get_logger("tools.tarin_xgb")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, required=True, help='gcfoest Net Model File')
    args = parser.parse_args()
    return args

def train_xgb(X_train, y_train, X_test, y_test):
    n_trees = 1000
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: n_trees={},X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        n_trees, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    clf = xgb.XGBClassifier(n_estimators=n_trees, max_depth=5, objective='multi:softprob',
            seed=0, silent=True, nthread=-1, learning_rate=0.1)
    eval_set = [(X_test, y_test)]
    clf.fit(X_train, y_train, eval_set=eval_set, eval_metric="merror", early_stopping_rounds=10)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)
    LOGGER.info('prec_xgb_{}={:.6f}%'.format(n_trees, prec*100.0))
    return clf, y_pred

if __name__ == "__main__":
    """
    Use xgboost to train and test the output produced by gcforest
    """
    args = parse_args()
    config = load_json(args.model)
    train_config = FGTrainConfig(config["train"])
    net = FGNet(config["net"], train_config.data_cache)

    data_train = get_dataset(config["dataset"]["train"])
    data_test = get_dataset(config["dataset"]["test"])

    train_xgb(
            concat_datas(net.get_outputs("train")), data_train.y, 
            concat_datas(net.get_outputs("test")), data_test.y)
    import IPython; IPython.embed()

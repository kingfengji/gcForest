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
from ..utils.log_utils import get_logger

LOGGER = get_logger('gcforest.estimators.est_utils')

def xgb_train(train_config, X_train, y_train, X_test, y_test):
    import xgboost as xgb
    LOGGER.info("X_train.shape={}, y_train.shape={}, X_test.shape={}, y_test.shape={}".format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    param = train_config["param"]
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    num_round = int(train_config["num_round"])
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    try:
        bst = xgb.train(param, xg_train, num_round, watchlist)
    except KeyboardInterrupt:
        LOGGER.info("Canceld by user's Ctrl-C action")
        return
    y_pred = np.argmax(bst.predict(xg_test), axis=1)
    acc = 100. * np.sum(y_pred == y_test) / len(y_test)
    LOGGER.info("accuracy={}%".format(acc))

# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
from sklearn.externals import joblib

from .base_estimator import BaseClassifierWrapper
from ..utils.log_utils import get_logger

LOGGER = get_logger('gcforest.estimators.sklearn_estimators')

def forest_predict_batch_size(clf, X):
    import psutil
    free_memory = psutil.virtual_memory().free
    if free_memory < 2e9:
        free_memory = int(2e9)
    max_mem_size = max(int(free_memory * 0.5), int(8e10))
    mem_size_1 = clf.n_classes_ * clf.n_estimators * 16
    batch_size = (max_mem_size - 1) / mem_size_1 + 1
    if batch_size < 10:
        batch_size = 10
    if batch_size >= X.shape[0]:
        return 0
    return batch_size

class SKlearnBaseClassifier(BaseClassifierWrapper):
    def _load_model_from_disk(self, cache_path):
        return joblib.load(cache_path)

    def _save_model_to_disk(self, clf, cache_path):
        joblib.dump(clf, cache_path)

class GCExtraTreesClassifier(SKlearnBaseClassifier):
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesClassifier
        super(GCExtraTreesClassifier, self).__init__(name, ExtraTreesClassifier, kwargs)
    
    def _default_predict_batch_size(self, clf, X):
        return forest_predict_batch_size(clf, X)

class GCRandomForestClassifier(SKlearnBaseClassifier):
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestClassifier
        super(GCRandomForestClassifier, self).__init__(name, RandomForestClassifier, kwargs)
    
    def _default_predict_batch_size(self, clf, X):
        return forest_predict_batch_size(clf, X)


class GCLR(SKlearnBaseClassifier):
    def __init__(self,name,kwargs):
        from sklearn.linear_model import LogisticRegression
        super(GCLR,self).__init__(name,LogisticRegression,kwargs)


class GCSGDClassifier(SKlearnBaseClassifier):
    def __init__(self,name,kwargs):
        from sklearn.linear_model import SGDClassifier
        super(GCSGDClassifier,self).__init__(name,SGDClassifier,kwargs)


class GCXGBClassifier(SKlearnBaseClassifier):
    def __init__(self,name,kwargs):
        import xgboost as xgb
        kwargs = kwargs.copy()
        if "random_state" in kwargs:
            kwargs["seed"] = kwargs["random_state"]
            kwargs.pop("random_state")
        super(GCXGBClassifier,self).__init__(name,xgb.XGBClassifier,kwargs)

"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
from .base_estimator import BaseClassifierWrapper
from .sklearn_estimators import GCExtraTreesClassifier, GCRandomForestClassifier
#from .xgb_estimator import GCXGBClassifier
from .kfold_wrapper import KFoldWrapper

def get_estimator_class(est_type):
    if est_type == "ExtraTreesClassifier":
        return GCExtraTreesClassifier
    if est_type == "RandomForestClassifier":
        return GCRandomForestClassifier
    #if est_type == "XGBClassifier":
    #    return GCXGBClassifier
    raise ValueError('Unkown Estimator Type, est_type={}'.format(est_type))

def get_estimator(name, est_type, est_args):
    est_class = get_estimator_class(est_type)
    return est_class(name, est_args)

def get_estimator_kfold(name, n_splits, est_type, est_args, random_state=None):
    est_class = get_estimator_class(est_type)
    return KFoldWrapper(name, n_splits, est_class, est_args, random_state=random_state)

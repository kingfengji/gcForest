"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
from .cifar10 import CIFAR10
from .ds_pickle import DSPickle
from .ds_pickle2 import DSPickle2
from .gtzan import GTZAN
from .imdb import IMDB
from .mnist import MNIST
from .olivetti_face import OlivettiFace
from .uci_adult import UCIAdult
from .uci_letter import UCILetter
from .uci_semg import UCISEMG
from .uci_yeast import UCIYeast

def get_ds_class(type_name):
    if type_name == 'cifar10':
        return CIFAR10
    if type_name == "ds_pickle":
        return DSPickle
    if type_name == "ds_pickle2":
        return DSPickle2
    if type_name == "gtzan":
        return GTZAN
    if type_name == 'imdb':
        return IMDB
    if type_name == 'mnist':
        return MNIST
    if type_name == "olivetti_face":
        return OlivettiFace
    if type_name == 'uci_adult':
        return UCIAdult
    if type_name == 'uci_letter':
        return UCILetter
    if type_name == 'uci_semg':
        return UCISEMG
    if type_name == 'uci_yeast':
        return UCIYeast
    return None

def get_dataset(ds_config):
    type_name = ds_config['type']
    ds_config.pop("type")
    ds_class = get_ds_class(type_name)
    if ds_class is None:
        raise ValueError('Unkonw Dataset Type: ', type_name)
    return ds_class(**ds_config)

gcForest v1.0
========
This is the official clone for the implementation of gcForest.

Package Official Website: http://lamda.nju.edu.cn/code_gcForest.ashx

This package is provided "AS IS" and free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@lamda.nju.edu.cn).

Description: A python 2.7 implementation of gcForest proposed in [1].                                             
A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code.  
The implementation is flexible enough for modifying the model or fit your own datasets.                           

Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks.               
            In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )                                                 

Requirements: This package is developed with Python 2.7, please make sure all the dependencies are installed,     
which is specified in requirements.txt                                                                            

ATTN: This package was developed and maintained by Mr.Ji Feng(http://lamda.nju.edu.cn/fengj/) .For any problem concerning the codes, please feel free to contact Mr.Feng.（fengj@lamda.nju.edu.cn)  or open some issues here.


package dependencies
========
The package is developed in python 2.7, higher version of python is not suggested for the current version.

run the following command to install dependencies before running the code:
```pip install -r requirements.txt```



Outline for README
========
* Package Overview
* Notes on Demo Scripts
* Notes on Model Specification Files
* Example
* Using Own Dataset

Package Overview
========
* lib/gcforest
    - code for the implementations for gcforest
* tools/train_fg.py
    - the demo script used for training Fine grained Layers
* tools/train_cascade.py
    - the demo script used for training Cascade Layers
* models/
    - folder to save models which can be used in tools/train_fg.py and tools/train_cascade.py
    - the gcForest structure is saved in json format
* logs
    - folder logs/gcforest is used to save the logfiles produced by demo scripts



Demo Scripts
=====
Here we give a brief discription on the args needed for demo scripts

tools/train_fg.py
-----------------
* --model: str
    - The config filepath for Fine grained models (in json format)
* --save_outputs: bool
    - if True. The output predictions produced by Fine Grained Model will be saved in model_cache_dir which is specified in  Model Config. This output will be used when Training Cascade Layer.
    - the default value is false

tools/train_cascade.py
----------------------
* --model: str
    - The model config filepath for cascade training (in json format)



Config Files
=================
Here we give a brief introduction on how to use model specification files, namely
* model specification for fine grained scanning stucture.
* model specification for cascade forests.

All the model specifications(in json files) are saved in  models/
For instance, all the model specification files needed for MNIST is stored in models/mnist/gcforest
* ca is short for cascade structure specifications
* fg is short for fine-grained structure specifications

You can define your own structure by writing similar json files.

FineGrained model's config (dataset)
------------------------
* dataset.train, dataset.test: [dict]
    - coresponds to the particular datasets defined in lib/datasets
    - type [str]: see lib/datasets/__init__.py for a reference
    - You can use your own dataset by writing similar wrappers.


FineGrained model's config (train)
----------------------------------

* train.keep_model_in_mem: [bool] default=0
    - if 0, the forest will be freed in RAM
* train.data_cache : [dict]
    - coresponds to the DataCache in lib/dataset/data_cache.py
* train.data_cache.cache_dir (str)
    - make sure to change "/mnt/raid/fengji/gcforest/cifar10/fg-tree500-depth100-3folds/datas" to your own path

FineGrained model's config (net)
----------------------------------
* net.outputs: [list]
    - List of the data names output by this model
* net.layers: [List of Layers]
    - Layer's Config, see lib/gcforest/layers for a reference

Cascade model's config (dataset)
------------------------------
Similar as FineGrained's model config (dataset)

Cascade model's config (cascade)
------------------------------
see lib/gcforest/cascade/cascade_classifier.py __init__  for a reference



Examples
========
Before runing the scripts, make sure to change

* train.data_cache.cache_dir in the Finegrained Model Config (eg: model/xxx/fg-xxxx.json)
* train.cascade.dataset.{train,test}.data_path in the Finegrained-Cascade Model Config (eg: model/xxx/fg-xxxx-ca.json)
* train.cascade.cascade.data_save_dir in the Finegrained Model Config (eg: model/xxx/ca-xxxx.json and model/xxx/fg-xxxx-ca.json)

To Train a gcForest(with fine grained scanning), you need to run two scripts.

* Fine Grained Scanning:  'tools/train_fg.py'
* Cascade Training: 'tools/train_cascade.py'




[UCI Letter](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition)
------------
* Get Data: you need to download the data by yourself by running the following command:
```Shell
cd dataset/uci_letter
sh get_data.sh
```
* Since we do not need to fine-grained scaning, we only train a Cascade Forest as follows:
    - `python tools/train_cascade.py --model models/uci_letter/gcforest/ca-tree500-n4x2-3folds.json --log_dir logs/gcforest/uci_letter/ca`

* UCI-Adult, YEAST can be trained with similar procedure.




MNIST
-----
* Get the data: The data will be automatically downloaded via 'lib/datasets/mnist.py', you do not need to do it yourself
* First Train the Fine Grained Forest:
    - Run `python tools/train_fg.py --model models/mnist/gcforest/fg-tree500-depth100-3folds.json --log_dir logs/gcforest/mnist/fg --save_outputs`
    - This means:
    1. Train a fine grained model for MNIST dataset,
    2. Using the structure defined in models/mnist/gcforest/fg-tree500-depth100-3folds.json
    3. save the log files in logs/gcforest/mnist/fg
    4. The output for the fine grained scanning predictions is saved in train.data_cache.cache_dir
* Then, train the cascade forest (Note: make sure you run the train_fg.py first)
    - run `python tools/train_cascade.py --model models/mnist/gcforest/fg-tree500-depth100-3folds-ca.json`
    - This means:
    1. Train the fine grained scaning results with cascade structure.
    2. The cascade model specification is defined in 'models/mnist/gcforest/fg-tree500-depth100-3folds-ca.json'
* You could also training a Cascade Forest without fine-grained scanning(but the accuracy will be much lower):
    - Run `python tools/train_cascade.py --model models/mnist/gcforest/ca-tree500-n4x2-3folds.json --log_dir logs/gcforest/mnist/ca`



[UCI sEMG](http://archive.ics.uci.edu/ml/datasets/sEMG+for+Basic+Hand+movements)
--------
* Get Data
```Shell
cd dataset/uci_semg
sh get_data.sh
```
* First Train the Fine Grained Forest:
    - `python tools/train_fg.py --model models/uci_semg/gcforest/fg-tree500-depth100-3folds.json --save_outputs --log_dir logs/gcforest/uci_semg/fg`
* Then, train the cascade forest (Note: make sure you run the train_fg.py first)
    - `python tools/train_cascade.py --model models/uci_semg/gcforest/fg-tree500-depth100-3folds-ca.json --log_dir logs/gcforest/uci_semg/gc`
* You could also training a Cascade Forest without fine-grained scanning(but the accuracy will be much lower):
    - `python tools/train_cascade.py --model models/uci_semg/gcforest/ca-tree500-n4x2-3folds.json --log_dir logs/gcforest/uci_semg/ca`


[GTZAN](http://marsyasweb.appspot.com/download/data_sets/)
-------
* Requirements(you need to install the following package)
librosa

* Get Data by yourself by running the following command
```Shell
cd dataset/gtzan
sh get_data.sh
cd ../..
python tools/audio/cache_feature.py --dataset gtzan --feature mfcc --split genre.trainval
```

* First Train the Fine Grained Forest:
    - `python tools/train_fg.py --model models/gtzan/gcforest/fg-tree500-depth100-3folds.json --save_outputs --log_dir logs/gcforest/gtzan/fg`
* Then, train the cascade forest (Note: make sure you run the train_fg.py first)
    - `python tools/train_cascade.py --model models/gtzan/gcforest/fg-tree500-depth100-3folds-ca.json --log_dir logs/gcforest/gtzan/gc`
* You could also training a Cascade Forest without fine-grained scanning(but the accuracy will be much lower):
    - `python tools/train_cascade.py --model models/gtzan/gcforest/ca-tree500-n4x2-3folds.json --log_dir logs/gcforest/gtzan/ca --save_outputs`

IMDB
----
* Cascade Forest:
    - `python tools/train_cascade.py --model models/imdb/gcforest/ca-tree500-n4x2-3folds.json --log_dir logs/gcforest/imdb/ca`

CIFAR10
-------
* First Train the Fine Grained Forest:
    - `python tools/train_fg.py --model models/cifar10/gcforest/fg-tree500-depth100-3folds.json --save_outputs`
* Then, train the cascade forest (Note: make sure you run the train_fg.py first)
    - `python tools/train_cascade.py --model models/cifar10/gcforest/fg-tree500-depth100-3folds-ca.json`


For You Own Datasets
========
* Data Format:
    0. Please refer lib/datasets/mnist.py as an example
    1. the dataset should has attribute X,y to represent the data and label
    2. y should be 1-d array
    3. For fine-grained scanning, X should be 4-d array (N x channel x H x W). (e.g. cifar10 shoud be Nx3x32x32, mnist should be Nx1x28x28, uci_semg should be Nx1x3000x1)
* Model Specifications:
    1. Save the json file in models/$dataset_name (recommended)
    2. for a detailed description, see section 'Config Files'
* If you only need to train a cascade forest, run tools/train_cascade.py.

Happy Hacking.

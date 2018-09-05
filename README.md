gcForest v1.1.1 Is Here!
========
This is the official clone for the implementation of gcForest.(The University's webserver is unstable sometimes, therefore we put the official clone here at github)

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





What's NEW:
========
* Scikit-Learn style API
* Some more detailed examples
* GPU support if you want to use xgboost as base estimators
* Support Python 3.5(v1.1.1)


v1.1.1 Python 3.5 Compatibility: The package should work for Python 3.5. Haven't check everything for now but it seems OK.


v1.1.1 Bug Fixed : When doing multiple predictions for the same model, the result will be consistant if you are using pooling layer. The bug only occurs for the scikit-learn APIs and now it is OK for the new api also.



Quick start
=====================

### The simplest way of using the library is as follows:
```
from gcforest.gcforest import GCForest
gc = GCForest(config) # should be a dict
X_train_enc = gc.fit_transform(X_train, y_train)
y_pred = gc.predict(X_test)
```
And that's it. Please see ```/examples/demo_mnist.py``` for a detailed useage.



For order versons AND some more model configs reported in the original paper, please refer:

* [v1.0](https://github.com/kingfengji/gcforest/tree/v1.0)



Supported Based Classifier
=====================
The based classifiers inside gcForest can be any classifiers. This library support the following ones:
* RandomForestClassifier
* XGBClassifier
* ExtraTreesClassifier
* LogisticRegression
* SGDClassifier



To add any classifiers, you could manually add them from ```lib/gcforest/estimators/__init__.py```



Define your own structure
=====================

### Define your model with a single json file.
* IF you only need cascading forest structure. You only need to write one json file. see /examples/demo_mnist-ca.json for a reference.(here -ca is for cascading)
* IF you need both fine grained and cascading forests, you will need to specifying the Finegraind structure of your model also.See /examples/demo_mnist-gc.json for a reference.
* Then, use gcforest.utils.config_utils.load_json to load your json file.

    ```
    config = load_json(your_json_file)
    gc = GCForest(config) # that's it
    ```
   and run ```python examples/demo_mnist.py --model examples/yourmodel.json```
### Define your model inside your python scripts.
  - You can also define the model structure inside your python script. The model config should be a python dictionary, see the ```get_toy_config``` in ```/examples/demo_mnist.py``` as a reference.





Supported APIs
=====================
*   ```fit_transform(X_train,y_train)```
*   ```fit_transform(X_train,y_train, X_test=X_test, y_test=y_test)```, this allows you to evaluate your model during training.
*   ```set_keep_model_in_mem(False)```. If your RAM is not enough, set this to false. (default is True). IF you set this to False, you would have to use ```fit_transform(X_train,y_train, X_test=X_test, y_test=y_test)``` to evaluate your model.
*   ```predict(X_test)```
*   ```transform(X_test)```


Supported Data Types
  =====================
  ### If you wish to use Cascade Layer only, the legal data type for X_train, X_test can be:
  *   2-D numpy array of shape (n_sampels, n_features).
  *   3-D or 4-D numpy array are also acceptable. For example, passing X_train of shape (60000, 28, 28) or (60000,3,28,28) will be automatically be reshape into (60000, 784)/(60000,2352).


  ### If you need to use Finegraind Layer, X_train, X_test MUST be a 4-D numpy array
  * for image-like data. the dimension should be (n_sampels, n_channels, n_height, n_width)
  * for sequence-like data. the dimension should be (n_sampels, n_features, seq_len, 1). e.g. For IMDB data, n_features is 1. For music MFCC data, n_features is 13.


Others
=====================
Please read ```examples/demo_mnist.py``` for a detailed walk-through.

package dependencies
========
The package is developed in python 2.7, higher version of python is not suggested for the current version.

run the following command to install dependencies before running the code:
```pip install -r requirements.txt```

Order Versions
=====================
For order versons, please refer:

* [v1.0](https://github.com/kingfengji/gcforest/tree/v1.0)

Happy Hacking.

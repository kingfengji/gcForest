# encoding: utf-8
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import gzip
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from .ds_base import ds_base

"""
X_train.len: min,mean,max=11,238,2494
X_test.len: min,mean,max=7,230,2315
"""
class IMDB(ds_base):
    def __init__(self, feature='tfidf', **kwargs):
        super(IMDB, self).__init__(**kwargs)
        if self.conf is not None:
            feature = self.conf.get('feature', 'tfidf')
        if feature.startswith('tfidf'):
            max_features = 5000
            (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
        else:
            (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=None, 
                    skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3)
        X, y = self.get_data_by_imageset(X_train, y_train, X_test, y_test)
        print('data_set={}, Average sequence length: {}'.format(self.data_set, np.mean(list(map(len, X)))))

        #feature
        if feature == 'origin':
            maxlen = 400
            X = sequence.pad_sequences(X, maxlen=maxlen)
        elif feature == 'tfidf':
            from sklearn.feature_extraction.text import TfidfTransformer
            transformer = TfidfTransformer(smooth_idf=False)
            #transformer = TfidfTransformer(smooth_idf=True)
            X_train_bin = np.zeros((len(X_train), max_features), dtype=np.int16)
            X_bin = np.zeros((len(X), max_features), dtype=np.int16)
            for i, X_i in enumerate(X_train):
                X_train_bin[i, :] = np.bincount(X_i, minlength=max_features)
            for i, X_i in enumerate(X):
                X_bin[i, :] = np.bincount(X_i, minlength=max_features)
            transformer.fit(X_train_bin)
            X = transformer.transform(X_bin)
            X = np.asarray(X.todense())
        elif feature == 'tfidf_seq':
            from sklearn.feature_extraction.text import TfidfTransformer
            transformer = TfidfTransformer(smooth_idf=False)
            maxlen = 400
            N = len(X)
            X_bin = np.zeros((N, max_features), dtype=np.int16)
            for i, X_i in enumerate(X):
                X_bin_i = np.bincount(X_i)
                X_bin[i, :len(X_bin_i)] = X_bin_i
            tfidf = transformer.fit_transform(X_bin)
            tfidf = np.asarray(tfidf.todense())
            X_id = sequence.pad_sequences(X, maxlen=maxlen)
            X = np.zeros(X_id.shape, dtype=np.float32)
            for i in range(N):
                X[i, :] = tfidf[i][X_id[i]]
        else:
            raise ValueError('Unkown feature: ', feature)

        X = X[:,np.newaxis,:,np.newaxis]
        self.X = self.init_layout_X(X)
        self.y = self.init_layout_y(y)

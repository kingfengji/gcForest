import numpy as np

from .cascade.cascade_classifier import CascadeClassifier
from .config import GCTrainConfig
from .fgnet import FGNet
from .utils.log_utils import get_logger

LOGGER = get_logger("gcforest.gcforest")


class GCForest(object):
    def __init__(self, config):
        self.config = config
        self.train_config = GCTrainConfig(config.get("train", {}))
        if "net" in self.config:
            self.fg = FGNet(self.config["net"], self.train_config.data_cache)
        else:
            self.fg = None
        if "cascade" in self.config:
            self.ca = CascadeClassifier(self.config["cascade"])
        else:
            self.ca = None

    def fit_transform(self, X_train, y_train, X_test=None, y_test=None, train_config=None):
        train_config = train_config or self.train_config
        if X_test is None or y_test is None:
            if "test" in train_config.phases:
                train_config.phases.remove("test")
            X_test, y_test = None, None
        if self.fg is not None:
            self.fg.fit_transform(X_train, y_train, X_test, y_test, train_config)
            X_train = self.fg.get_outputs("train")
            if "test" in train_config.phases:
                X_test = self.fg.get_outputs("test")
        if self.ca is not None:
            _, X_train, _, X_test, _, = self.ca.fit_transform(X_train, y_train, X_test, y_test, train_config=train_config)

        if X_test is None:
            return X_train
        else:
            return X_train, X_test

    def transform(self, X):
        """
        return:
            if only finegrained proviede: return the result of Finegrained
            if cascade is provided: return N x (n_trees in each layer * n_classes)
        """
        if self.fg is not None:
            X = self.fg.transform(X)
        y_proba = self.ca.transform(X)
        return y_proba

    def predict_proba(self, X):
        if self.fg is not None:
            X = self.fg.transform(X)
        y_proba = self.ca.predict_proba(X)
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def set_data_cache_dir(self, path):
        self.train_config.data_cache.cache_dir = path

    def set_keep_data_in_mem(self, flag):
        """
        flag (bool):
            if flag is 0, data will not be keeped in memory.
            this is for the situation when memory is the bottleneck
        """
        self.train_config.data_cache.config["keep_in_mem"]["default"] = flag

    def set_keep_model_in_mem(self, flag):
        """
        flag (bool):
            if flag is 0, model will not be keeped in memory.
            this is for the situation when memory is the bottleneck
        """
        self.train_config.keep_model_in_mem = flag

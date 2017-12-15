from .data_cache import DataCache


class GCTrainConfig(object):
    def __init__(self, train_config):
        self.keep_model_in_mem = train_config.get("keep_model_in_mem", True)
        self.random_state = train_config.get("random_state", 0)
        self.model_cache_dir = strip(train_config.get("model_cache_dir", None))
        self.data_cache = DataCache(train_config.get("data_cache", {}))
        self.phases = train_config.get("phases", ["train", "test"])

        for data_name in ("X", "y"):
            if data_name not in self.data_cache.config["keep_in_mem"]:
                self.data_cache.config["keep_in_mem"][data_name] = True
            if data_name not in self.data_cache.config["cache_in_disk"]:
                self.data_cache.config["cache_in_disk"][data_name] = False


def strip(s):
    if s is None:
        return None
    s = s.strip()
    if len(s) == 0:
        return None
    return s

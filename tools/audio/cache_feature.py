"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
from joblib import Parallel, delayed
import librosa
import numpy as np
import os, os.path as osp
import sys

sys.path.insert(0, 'lib')
from gcforest.datasets.gtzan import GTZAN
from gcforest.utils.audio_utils import select_feature_func

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='gtzan',
            choices=['gtzan'])
    parser.add_argument('--split', dest='split', type=str, required=True)
    parser.add_argument('--feature', dest='feature', type=str, required=True)
    args = parser.parse_args()
    return args

def save_cache(src_path, des_path, get_feature_func):
    des_path = osp.splitext(des_path)[0] + '.npy'
    try:
        X, sr = librosa.load(src_path)
        src = int(sr)
        feature = get_feature_func(X, sr)
        print('[INFO] Saving Cache in {} ...'.format(des_path))
        des_par = osp.abspath(osp.join(des_path, osp.pardir))
        if not osp.exists(des_par):
            os.makedirs(des_par)
    except Exception, e:
        print("[ERROR] Unkown error happend when dealing with{}".format(src_path))
        #print(e)
        return -1
    np.save(des_path, feature)
    return 0

if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'urbansound8k':
        dataset = UrbanSound8K(data_set=args.split, layout_x='rel_path')
    elif args.dataset == 'gtzan':
        dataset = GTZAN(data_set=args.split, layout_x='rel_path')

    feature_name = args.feature
    get_feature_func = select_feature_func(feature_name)

    rel_paths = dataset.X
    src_paths = []
    des_paths = []
    for rel_path in rel_paths:
        des_path = osp.join(dataset.cache_base, feature_name, rel_path)
        if osp.exists(des_path):
            continue
        src_paths.append(osp.join(dataset.data_base, rel_path))
        des_paths.append(des_path)
    print('Total={}, Done={}, Undo={}'.format(len(rel_paths), len(rel_paths)-len(src_paths), len(src_paths)))
    print('src_paths[:5]={}'.format(src_paths[:5]))
    print('des_paths[:5]={}'.format(des_paths[:5]))

    status = Parallel(n_jobs=-1, verbose=1, backend='multiprocessing')(
            delayed(save_cache)(src_paths[i], des_paths[i], get_feature_func)
            for i, src_path in enumerate(src_paths))

    # check error
    error_src_paths = []
    for i, src_path in enumerate(src_paths):
        if status[i] == -1:
            error_src_paths.append(src_path)
    print('len(error_src_paths)={}, error_src_paths[:5]={}'.format(
        len(error_src_paths), error_src_paths[:5]))
    if len(error_src_paths) > 0:
        error_save_path = dataset.data_set_path + '.error'
        with open(error_save_path, 'wb') as f:
            for error_src_path in error_src_paths:
                f.write('{}\n'.format(error_src_path))
    import IPython; IPython.embed()

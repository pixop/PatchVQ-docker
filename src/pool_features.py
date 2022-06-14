#import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
import joblib.parallel
from functools import partial
import math
from pathlib import Path
import torch.nn as nn
import pandas as pd
from loguru import logger
import numpy as np
import torch

def takespread_idx(N, num, clip_size=16):
    length = float(N-clip_size)
    for i in range(num):
        start = int(math.ceil(i * length / (num-1)))
        yield start, start+clip_size

def load_feature(vid, feat, path):
    npy_file = Path(path)/feat/(str(vid) + '.npy')
    with open(npy_file, 'rb') as f:
        features = np.load(f)
    return torch.Tensor(features)

def save_feature(x, vid, feat, path):
    npy_file =  Path(path)/feat/(str(vid) + '.npy')
    Path(npy_file).parent.mkdir(parents=True, exist_ok=True)
    with open(npy_file, 'wb') as f:
        np.save(f, x)

class FeatPooler():
    fn_col = 'name'
    frame_num_col = 'frame_number'

    def get_df(self, path):
        return  pd.read_csv(path).set_index(self.fn_col)

    def __init__(self, d, name, input_suffix, output_suffix, pool_size = 16, clip_num=16):
        self.path = Path(d['dir'])
        self.fn_col = d['fn_col']
        if 'frame_num_col' in d.keys():
            self.frame_num_col = d['frame_num_col']
        self.roi_index = [0, 1, 0, 2]
        self.m = nn.AdaptiveAvgPool1d(pool_size)
        self.df = self.get_df(self.path/d['csv_labels'])
        self.input_feat = name + input_suffix
        self.output_feat = name + output_suffix
        self.clip_num = clip_num

    @classmethod
    def from_dict(cls, d, **kwargs):
        return cls(d, **kwargs)

    def prepare_feat(self, vid):
        logger.info(f'pooling : {vid}')
        df = self.df
        _feats = load_feature(vid, self.input_feat, self.path/'features')
        if _feats.dim() == 4:
            _feats = _feats.unsqueeze(dim=1)

        feats = _feats.refine_names('N', 'roi', 'C', 'H', 'W')
        feats = feats.flatten(['C', 'H', 'W'], 'features')
        #feats = feats.flatten(start, 'features')
        feats = feats.align_to('roi', 'features', 'N') # N C L

        logger.debug(f'preparing:   {_feats.shape} to {feats.shape}')
        return feats

    def __call__(self, vid): # pool_feat
        feats = self.prepare_feat(vid)
        n_frames = feats.shape[-1]
        pooled_feat_list = [] # 16 clips, indexes
        for start, end in takespread_idx(n_frames, self.clip_num):  #n_chunks_idx(n_frames, self.clip_num):
            feat = feats[ :,  :, start:end].rename(None)
            pooled_feat = self.m(feat) # [1, features, frame_num] --> [1, features, clip_num]
            pooled_feat_list.append(pooled_feat.permute(0,2,1)) # --> [1, clip_num, features]

        # clip_num x -1  x  features
        pooled_feats = torch.cat(pooled_feat_list, dim=0).view(-1, pooled_feat_list[0].shape[-1])
        logger.debug(f'%cat {pooled_feat_list[0].shape} with length {len(pooled_feat_list)} to {pooled_feats.shape}')
        save_feature(pooled_feats, vid, self.output_feat, self.path/'features')  # --> [1, clip_num, features]
        logger.debug(f'done:   {pooled_feats.shape} ')
        return pooled_feats


#import multiprocessing
#from joblib import Parallel, delayed

def pool_features(database, name, input_suffix="", output_suffix="_pooled", pool_size=16):
  pooler = FeatPooler.from_dict(database, name=name, input_suffix=input_suffix, output_suffix=output_suffix, pool_size=pool_size)

  vids = pooler.df.index.tolist()
  #num_cores = multiprocessing.cpu_count()
  vids_todo = [vid for vid in vids if not (pooler.path/'features'/(pooler.output_feat + '/' + str(vid) + '.npy')).exists()]

  desc = f'{name}({input_suffix} --> {output_suffix})'

  #with tqdm(desc=desc, total=len(vids_todo)) as progress_bar:
  #[pooler(vid) for vid in vids_todo]
  for vid in vids_todo:
    pooler(vid)

  # parallel:
  # with tqdm_joblib(tqdm(desc=desc, total=len(vids_todo))) as progress_bar:
  #     Parallel(n_jobs=num_cores)(delayed(pooler)(vid) for vid in vids_todo)

database = {
    "__name__": "Bitmovin",
    "dir": "data/Bitmovin/",
    "csv_labels": "labels.csv",
    "fn_col": "name",
    "label_col": "mos"
    }
pool_features(database, "paq2piq", input_suffix="", output_suffix="_pooled", pool_size=16)
pool_features(database, "r3d_18", input_suffix="", output_suffix="_pooled", pool_size=16)


import os.path, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import logging

import warnings

from fastiqa.vqa import *

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

# modify the json files to point to your database locations
LSVQ  = load_json('json/LSVQ.json')
LIVE_VQC_tiny = {
    "__name__": "inference",
    "dir": "data/inference/",
    "csv_labels": "labels.csv",
    "fn_col": "name",
    "label_col": "mos"
    }

n_epoch = 10
bs = 128

class PVQ(InceptionTimeModel):
    siamese=True

    def bunch(self, dls):
        if isinstance(dls, dict):
            # download extracted features or extract by your own
            feats = FeatureBlock('paq2piq_pooled', roi_index=None, clip_num=None, clip_size=None), \
            FeatureBlock('r3d_18_pooled', roi_index=None, clip_num=None, clip_size=None)
            dls = Feat2MOS.from_dict(dls, bs=bs, feats=feats)

        if not isinstance(dls.label_col, (list, tuple)): # database contains no patch labels
            print(f'set self.n_out = 1')
            self.n_out = 1
        return dls

e = IqaExp('release', gpu=0)
e += IqaLearner(dls=LSVQ, model = PVQ(c_in=2048+2048, n_out=4), loss_func=L1LossFlat())

# train the model if pretrained model is not available
e.fit_one_cycle(n_epoch)

# load the trained model
e.load()

# perform MOS inference
warnings.filterwarnings('ignore')
print(e.mos_inference([All(LIVE_VQC_tiny)], cache=False))

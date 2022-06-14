import os.path, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from fastiqa.vqa import *

# modify the json files to point to your database locations
LSVQ  = load_json('json/LSVQ.json')
LIVE_VQC_tiny = {
    "__name__": "Bitmovin",
    "dir": "data/Bitmovin/", #"/content/PatchVQ/data/LIVE_VQC_tiny/",
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

# cross database validation
print(e.valid([All(LIVE_VQC_tiny)], cache=False))

# combine filenames and scores for LIVE_VQC predictions
df1 = pd.read_csv('release/data/LIVE_VQC/labels.csv')
df1['full_output'] = pd.read_csv('release/PVQ/train@LSVQ/default/valid@LIVE_VQC_all.csv')['output']
df1['full_target'] = pd.read_csv('release/PVQ/train@LSVQ/default/valid@LIVE_VQC_all.csv')['target']

# combine filenames and scores for LIVE_VQC_tiny predictions
df2 = pd.read_csv('data/LIVE_VQC_tiny/labels.csv')
df2['tiny_output'] = pd.read_csv('release/PVQ/train@LSVQ/default/valid@LIVE_VQC_tiny_all.csv')['output']
df2['tiny_target'] = pd.read_csv('release/PVQ/train@LSVQ/default/valid@LIVE_VQC_tiny_all.csv')['target']

# only select those in the tiny set
df1 = df1[df1['name'].isin(df2['name'])]

# add tiny_output and tiny_target
df1.reset_index(inplace=True)
df1['tiny_output'] = df2['tiny_output']
df1['tiny_target'] = df2['tiny_target']

# swap full_target and tiny_output
def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

df1 = df_column_switch(df1, 'full_target', 'tiny_output')

# check if the targets are equal
test_eq(df1.full_target, df2.tiny_target)

# check if the predictions are equal
#test_eq(df1.full_output, df2.tiny_output)

#print(df1)
#print(df2)


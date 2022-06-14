import os.path, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from fastiqa.vqa import *
from fastiqa.log import *
from torchvision.models.video.resnet import *
from tqdm import tqdm

def get_features(x, name, bs, vid_id):
    try:
        x.dls.bs = bs
        x.dls.set_vid_id(vid_id)
        x.extract_features(name=name, skip_exist=True)
    except RuntimeError:
        print('WTF')
        tmp_bs = bs
        while tmp_bs > 1:
            tmp_bs //= 2
            try:
                x.dls.bs = tmp_bs
                x.extract_features(name=name, skip_exist=True)
                break
            except RuntimeError:
                print(f'CUDA out of memory. Reduce bs from {bs} to {tmp_bs}.')
                continue

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

bs = 128 
clip_num = None
clip_size = 1
roi_col = None

name = 'paq2piq'

dls = SingleVideo2MOS.from_dict({
    "__name__": "inference",
    "dir": "data/inference/", # change the location accordingly
    "csv_labels": "labels.csv",
    "fn_col": "name",
    "label_col": "mos"
    },
    use_nan_label=True, clip_num=clip_num, clip_size=clip_size,
    bs=bs)
dls.roi_col = roi_col

path_to_model_state = './RoIPoolModel-fit.10.bs.120.pth' # change the location accordingly
model = LegacyRoIPoolModel(backbone=resnet18, pool_size=(2,2))
model_state = torch.load(path_to_model_state, map_location=lambda storage, loc: storage)
model.load_state_dict(model_state["model"])

learn = TestLearner(dls, model)
vid_list = dls.video_list.index.tolist() 
learn.dls.set_vid_id(vid_list[0])

e = IqaExp('exp_features', gpu=0, seed=None)
e[name] = learn
e.run(lambda x: [get_features(x, name, bs=bs, vid_id=vid_id) for vid_id in tqdm(vid_list)])

#import click
from fastiqa.vqa import *
#from fastiqa.models.mobilenet.v2 import mobilenet3d_v2
#from torchvision.models import *

from fastiqa.log import *
from torchvision.models.video.resnet import *
from tqdm import tqdm

def get_features(x, name, bs, vid_id):
    try:
        x.dls.bs = bs
        x.dls.set_vid_id(vid_id)
        x.extract_features(name=name, skip_exist=True)
    except RuntimeError:
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

bs = 8 
clip_num = 40
clip_size = 8
roi_col = None

def r3d18_K_200ep(pretrained=True, **kwargs):
    model = resnet.generate_model(model_depth=18, n_classes=700, **kwargs)
    PATH =  'fastai-r3d18_K_200ep.pth' # <---- change the path accordingly
    if pretrained:
        model.load_state_dict(torch.load(PATH))
    return model

model = RoIPoolModel(backbone=r3d18_K_200ep)
name = 'r3d_18'

dls = SingleVideo2MOS.from_dict({
    "__name__": "Bitmovin",
    "dir": "data/Bitmovin/",
    "csv_labels": "labels.csv",
    "fn_col": "name",
    "label_col": "mos"
    },
    use_nan_label=True, clip_num=clip_num, clip_size=clip_size,
    bs=bs)
dls.roi_col = roi_col

learn = TestLearner(dls, model)
vid_list = dls.video_list.index.tolist() 
learn.dls.set_vid_id(vid_list[0])

e = IqaExp('exp_features', gpu=0, seed=None)
e[name] = learn
e.run(lambda x: [get_features(x, name, bs=bs, vid_id=vid_id) for vid_id in tqdm(vid_list)])

feats = load_feature(vid='G005', feat='r3d_18', path='data/LIVE_VQC_tiny/features')
print(feats.shape)

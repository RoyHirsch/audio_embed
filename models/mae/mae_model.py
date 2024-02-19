import torch
import models_mae

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)(in_chans=1, audio_exp=True, img_size=(1024, 128))
    # load model
    # checkpoint = torch.load(chkpt_dir, map_location='cpu')
    # msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

chkpt_dir = '/checkpoint/berniehuang/experiments/53636601/checkpoint-19.pth'
model = prepare_model(chkpt_dir, 'mae_vit_base_patch16')

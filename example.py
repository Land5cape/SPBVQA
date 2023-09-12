import os
import torch
from torch.utils.data import DataLoader

from eval import val_model
from modules.dataset import MyDataset
from modules.loss import MyHuberLoss
from modules.model import mynet18
from modules.utils import save_crop_video, label_MOS

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    my_model = mynet18()
    my_model.to(device)
    criterion = MyHuberLoss()

    model_load_path = 'model-save/SPBVQA.pth'

    video_path = 'videos/9726323401_original_centercrop_960x540_8s.mp4'
    video_save_dir = 'videos/9726323401_original_centercrop_960x540_8s'
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)

    label_dict = {
        '9726323401_original_centercrop_960x540_8s.mp4': 3.272
    }
    MOS_range = [1.22, 4.64]
    tail = '.mp4'

    print('preparing')
    save_crop_video(video_path, video_save_dir, tail)

    model = torch.load(model_load_path)
    my_model.load_state_dict(model['net'])

    val_dataset = MyDataset(video_save_dir, label_dict, MOS_range, tail)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    print('predicting')
    pred, label = val_model(my_model, device, criterion, val_loader, MOS_range, show_detail=False)
    print('normalized predict score:', pred, 'normalized MOS:', label)

    pred = label_MOS(pred, MOS_range)
    label = label_MOS(label, MOS_range)
    print('predict score:', pred, 'MOS:', label)

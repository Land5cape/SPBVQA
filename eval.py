import torch
import numpy as np
from modules.utils import get_PLCC, get_RMSE, get_SROCC, mos_scatter, get_KROCC, save_pred_label


def val_model(my_model, device, criterion, val_loader, MOS_range, show_detail=True):
    my_model.eval()
    with torch.no_grad():

        val_pred = {}
        val_label = {}
        val_epoch_loss = 0
        for i, inputs in enumerate(val_loader):

            frames = inputs['frames'].to(device)
            label = inputs['label'].to(device)
            file_name = inputs['file_name']

            output = my_model(frames)
            loss = criterion(output, label)  # 算损失

            val_epoch_loss += loss.item()
            l = label.view(-1).cpu().numpy()  # 一维array
            p = output[:, -1].cpu().detach().numpy()

            for j, name in enumerate(file_name):
                if name not in val_pred.keys():
                    val_pred[name] = []
                    val_label[name] = l[j]
                val_pred[name].append(p[j])

            if show_detail:
                print('\033[1;33m'
                      '----------------------------------------'
                      'Validating: %d / %d'
                      '----------------------------------------'
                      ' \033[0m' % (i + 1, len(val_loader)))
                print('\033[1;31m loss: ', loss.item(), '\033[0m')
                # print('\033[1;34m label: ', label.view(-1).data, '\033[0m')
                # print('\033[1;34m predict: ', output[:, -1].data, '\033[0m')
                # print('output: ', output)

        video_val_pred = []
        video_val_label = []
        for d in val_pred.keys():
            video_val_label.append(val_label[d])
            video_val_pred.append(np.mean(val_pred[d]))

        video_val_label = np.array(video_val_label)
        video_val_pred = np.array(video_val_pred)

        if video_val_pred.shape[0] > 1:

            val_rmse = get_RMSE(video_val_pred, video_val_label, MOS_range)
            val_plcc = get_PLCC(video_val_pred, video_val_label)
            val_srocc = get_SROCC(video_val_pred, video_val_label)
            val_krocc = get_KROCC(video_val_pred, video_val_label)
            val_loss = val_epoch_loss / (i + 1)

            print(val_loss, val_rmse, val_plcc, val_srocc, val_krocc)

            fig = mos_scatter(video_val_pred, video_val_label, show_fig=False)

            return val_loss, val_rmse, val_plcc, val_srocc, val_krocc, fig, video_val_pred, video_val_label
        else:
            return video_val_pred[0], video_val_label[0]
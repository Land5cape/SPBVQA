import csv
import os
import random

import numpy as np
from cv2 import CAP_PROP_FRAME_HEIGHT, resize, CAP_PROP_FRAME_COUNT, VideoCapture, CAP_PROP_FRAME_WIDTH, \
    CAP_PROP_POS_FRAMES
import skvideo.io
import torch
import torch.nn.init as init
import torch.nn as nn
from scipy import stats

from opts import INPUT_LENGTH, INPUT_SIZE, CLIP_STRIDE, SEG_STRIDE, RESIZE_SIZE, RESIZE

# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_pred_label(csv_path, list1, list2):
    infos = zip(list1, list2)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for info in infos:
            writer.writerow(info)


def adjust_learning_rate(optimizer, epoch, step, rate, min_lr):
    lr = optimizer.param_groups[0]['lr']

    if lr <= min_lr:
        return

    if epoch % step == step - 1:
        cur_lr = lr * rate
        new_lr = cur_lr if cur_lr > min_lr else min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def weigth_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def MOS_label(MOS, MOS_range):
    MOS_min, MOS_max = MOS_range
    label = (MOS - MOS_min) / (MOS_max - MOS_min)
    return label


def label_MOS(label, MOS_range):
    MOS_min, MOS_max = MOS_range
    MOS = label * (MOS_max - MOS_min) + MOS_min
    return MOS


def get_PLCC(y_pred, y_val):
    return stats.pearsonr(y_pred, y_val)[0]


def get_SROCC(y_pred, y_val):
    return stats.spearmanr(y_pred, y_val)[0]


def get_KROCC(y_pred, y_val):
    return stats.stats.kendalltau(y_pred, y_val)[0]


def get_RMSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.sqrt(np.mean((y_p - y_v) ** 2))


def get_MSE(y_pred, y_val, MOS_range):
    y_p = label_MOS(y_pred, MOS_range)
    y_v = label_MOS(y_val, MOS_range)
    return np.mean((y_p - y_v) ** 2)


def mos_scatter(pred, mos, show_fig=False):
    fig = plt.figure()
    plt.scatter(mos, pred, s=5, c='g', alpha=0.5)
    plt.xlabel('MOS')
    plt.ylabel('PRED')
    plt.plot([0, 1], [0, 1], linewidth=0.5)
    if show_fig:
        plt.show()
    return fig


def fig2data(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def read_video_npy(video_path, start=None, intput_length=None):
    video_frames = np.load(video_path, mmap_mode='r')
    if start and intput_length:
        video_frames = video_frames[start: start + intput_length]
    frames = torch.from_numpy(video_frames)
    frames = frames.permute([3, 0, 1, 2])  # 维度变换 (3, time, height, width)
    frames = frames.float()
    frames = frames / 255  # 输入归一化
    return frames


def read_video(video_path, start, input_length):
    cap = VideoCapture()
    cap.open(video_path)
    if not cap.isOpened():
        raise Exception("VideoCapture failed!")

    video_frames = np.zeros((input_length, int(cap.get(CAP_PROP_FRAME_HEIGHT)),
                             int(cap.get(CAP_PROP_FRAME_WIDTH)), 3), dtype='uint8')

    cap.set(CAP_PROP_POS_FRAMES, start)

    for i in range(input_length):
        rval, frame = cap.read()  # h,w,c
        if rval:
            video_frames[i] = frame
        else:
            raise Exception("VideoCapture failed!")
    cap.release()
    frames = torch.from_numpy(video_frames)
    frames = frames.permute([3, 0, 1, 2])  # 维度变换 (3, time, height, width)
    frames = frames.float()
    frames = frames / 255  # 输入归一化

    return frames


def resize_frame(frame, size):
    h, w, _ = frame.shape
    if (h <= w and h == size) or (w <= h and w == size):
        return frame
    if h < w:
        oh = size
        ow = int(size * w / h)
    else:
        ow = size
        oh = int(size * h / w)
    return resize(frame, (oh, ow))


def save_crop_video(video_path, video_save_dir, tail):
    if not os.path.exists(video_save_dir):
        os.mkdir(video_save_dir)
    video_name = os.path.basename(video_path)

    if tail == '.yuv':
        video_frames = skvideo.io.vread(video_path, 1080, 1920,
                                        inputdict={'-pix_fmt': 'yuvj420p'})
        length = video_frames.shape[0]
        if RESIZE:
            frames = []
            for i in range(length):
                frames.append(resize_frame(video_frames[i, :, :, :], RESIZE_SIZE))
            video_frames = frames
    else:
        cap = VideoCapture()
        cap.open(video_path)
        if not cap.isOpened():
            raise Exception("VideoCapture failed!")

        length = int(cap.get(CAP_PROP_FRAME_COUNT))
        video_frames = []

        for i in range(length):
            rval, frame = cap.read()
            if rval:
                if RESIZE:
                    frame = resize_frame(frame, RESIZE_SIZE)
                video_frames.append(frame)
            else:
                continue
        cap.release()

    video_frames = np.array(video_frames, dtype='uint8')
    H = video_frames.shape[1]
    W = video_frames.shape[2]

    n_clips = count_clip(length, CLIP_STRIDE, SEG_STRIDE)
    print(video_frames.shape, n_clips)

    _save = np.zeros((INPUT_LENGTH, INPUT_SIZE, INPUT_SIZE, 3), dtype='uint8')

    hIdxMax = H - INPUT_SIZE
    wIdxMax = W - INPUT_SIZE

    if hIdxMax < 0:
        return
    if wIdxMax < 0:
        return

    hIdx = [INPUT_SIZE * i for i in range(0, hIdxMax // INPUT_SIZE + 1)]
    wIdx = [INPUT_SIZE * i for i in range(0, wIdxMax // INPUT_SIZE + 1)]

    if hIdxMax % INPUT_SIZE != 0 and H % INPUT_SIZE > (INPUT_SIZE // 4):
        hIdx.append(hIdxMax)
    if wIdxMax % INPUT_SIZE != 0 and W % INPUT_SIZE > (INPUT_SIZE // 4):
        wIdx.append(wIdxMax)

    count = 0
    for clip in range(n_clips):
        for seg in range(SEG_STRIDE):
            pos = int(clip * CLIP_STRIDE) + seg
            for h in hIdx:
                for w in wIdx:
                    video_save_path = os.path.join(video_save_dir, video_name + '_{}_{}_{}.npy'.format(pos, h, w))
                    if os.path.exists(video_save_path):
                        continue
                    for idx in range(INPUT_LENGTH):
                        _save[idx] = video_frames[pos + idx * SEG_STRIDE, h:h + INPUT_SIZE, w:w + INPUT_SIZE, :]
                    np.save(video_save_path, _save)
                    count += 1
    print(count)


def get_video_name(file_name, tail):
    video_name = os.path.basename(file_name)
    if not video_name.endswith('.npy'):
        return video_name
    else:
        return video_name.split(tail)[0] + tail


def get_encoding(file, filelen):
    import chardet
    with open(file, 'rb') as f:
        tmp = chardet.detect(f.read(filelen))
    return tmp['encoding']


def read_label(csv_path, id1, id2, tail):
    dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:
            name = row[id1]
            if not name.endswith(tail):
                name = name + tail
            dict[name] = float(row[id2])
    return dict


def read_split(csv_path):
    info_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:  # name, class
            info_dict[row[0]] = row[1]
    return info_dict


def write_split(csv_path, infos, wtype):
    with open(csv_path, wtype, encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if wtype == 'w':
            writer.writerow(['file_name', 'class'])
        for info in infos:
            writer.writerow(info)


def count_clip(n_frames, clip_stride, seg_stride):
    clip_length = INPUT_LENGTH * seg_stride
    tIdMax = n_frames - clip_length
    n_clips = tIdMax // clip_stride + 1
    return int(n_clips)


def count_block(video_path, block_stride):
    vc = VideoCapture()
    vc.open(video_path)
    if not vc.isOpened():
        raise Exception("VideoCapture failed!")
    H = vc.get(CAP_PROP_FRAME_HEIGHT)
    W = vc.get(CAP_PROP_FRAME_WIDTH)

    hIdxMax = H - INPUT_SIZE
    wIdxMax = W - INPUT_SIZE
    hNums = hIdxMax // block_stride + 1 if hIdxMax % block_stride == 0 \
        else hIdxMax // block_stride + 2
    wNums = wIdxMax // block_stride + 1 if wIdxMax % block_stride == 0 \
        else wIdxMax // block_stride + 2

    return int(hNums), int(wNums)


def data_split(full_list, ratio, shuffle=True):
    nums_total = len(full_list)
    offset1 = int(nums_total * ratio[0])
    offset2 = int(nums_total * (ratio[0] + ratio[1]))

    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3


def cal(csv_path, d):
    KoNViD_1k_MOS = [1.22, 4.64]
    CVD2014_MOS = [-6.50, 93.38]
    LiveQ_MOS = [16.5621, 73.6428]
    LiveV_MOS = [6.2237, 94.2865]
    UGC_MOS = [1.242, 4.698]

    if d == 1:
        mos = KoNViD_1k_MOS
    elif d == 2:
        mos = CVD2014_MOS
    elif d == 4:
        mos = LiveV_MOS
    elif d == 5:
        mos = UGC_MOS

    l1 = []
    l2 = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:
            l1.append(float(row[0]))
            l2.append(float(row[1]))

    l1 = np.array(l1)
    l2 = np.array(l2)

    plcc = get_PLCC(l1, l2)
    srocc = get_SROCC(l1, l2)
    krocc = get_KROCC(l1, l2)
    rmse = get_RMSE(l1, l2, mos)

    print(srocc, krocc, plcc, rmse)


if __name__ == '__main__':
    a = label_MOS(0.6,  [1.22, 4.64])
    print(a)
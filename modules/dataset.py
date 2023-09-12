import os
import time

import torch
from torch.utils.data import Dataset
from modules.utils import get_video_name, write_split, read_split, data_split, read_label, MOS_label, \
    save_crop_video, read_video_npy
from opts import ds


class Preprocesser():
    def __init__(self, video_dir, label_path, split_path,
                 ratio=None, preprocess=False, tail='.mp4'):

        print("\033[1;34m video_dir: ", video_dir, "\033[0m")
        self.tail = tail

        if ds == 1:
            self.label_dict = read_label(label_path, 2, 3, tail)
        elif ds == 2:
            self.label_dict = read_label(label_path, 1, 3, tail)
        elif ds == 3 or ds == 4:
            self.label_dict = read_label(label_path, 0, 1, tail)
        elif ds == 5:
            self.label_dict = read_label(label_path, 0, 4, '_crf_10_ss_00_t_20.0' + tail)

        self.video_train_dir = os.path.join(video_dir, 'train')
        self.video_val_dir = os.path.join(video_dir, 'val')
        self.video_test_dir = os.path.join(video_dir, 'test')

        video_paths = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                video_path = os.path.join(root, file)
                if video_path.endswith(self.tail):
                    video_paths.append(video_path)

        cur_video_paths = []
        for i in video_paths:
            if get_video_name(i, tail) in self.label_dict.keys():
                cur_video_paths.append(i)
        self.video_paths = cur_video_paths

        print("\033[1;34m video num: ", len(self.video_paths), "\033[0m")

        if preprocess:
            if not ratio:
                raise Exception("Set ratio!")
            self.train_paths, self.val_paths, self.test_paths = \
                data_split(self.video_paths, ratio, shuffle=True)
            self.write_infos(split_path)
            self.split_dict = read_split(split_path)
        else:
            self.train_paths, self.val_paths, self.test_paths = [], [], []
            self.split_dict = read_split(split_path)
            for vp in self.video_paths:
                vn = get_video_name(vp, self.tail)
                if self.split_dict[vn] == 'train':
                    self.train_paths.append(vp)
                elif self.split_dict[vn] == 'val':
                    self.val_paths.append(vp)
                else:
                    self.test_paths.append(vp)

    def write_infos(self, path):
        infos = []
        for vp in self.train_paths:
            vn = get_video_name(vp, self.tail)
            infos.append([vn, 'train'])
        for vp in self.val_paths:
            vn = get_video_name(vp, self.tail)
            infos.append([vn, 'val'])
        for vp in self.test_paths:
            vn = get_video_name(vp, self.tail)
            infos.append([vn, 'test'])
        write_split(path, infos, 'w')

    def crop_video_blocks(self):
        print('crop video blocks!')
        time.sleep(5)

        for i, vp in enumerate(self.video_paths):
            vn = get_video_name(vp, self.tail)
            print(i, vn)
            if self.split_dict[vn] == 'train':
                video_save_dir = self.video_train_dir
            elif self.split_dict[vn] == 'val':
                video_save_dir = self.video_val_dir
            else:
                video_save_dir = self.video_test_dir
            save_crop_video(vp, video_save_dir, self.tail)


class MyDataset(Dataset):

    def __init__(self, video_dir, label_dict, MOS_range, tail):
        super(MyDataset, self).__init__()

        self.video_paths = []
        self.video_names = []
        self.video_labels = []

        for x in os.scandir(video_dir):
            if x.name.endswith('.npy'):
                self.video_paths.append(x.path)
                n = get_video_name(x.path, tail)
                self.video_names.append(n)
                l = torch.tensor(label_dict[n]).view(-1)
                self.video_labels.append(MOS_label(l, MOS_range))

    def __getitem__(self, index):

        file_path = self.video_paths[index]
        file_name = self.video_names[index]
        label = self.video_labels[index]
        frames = read_video_npy(file_path)

        sample = {
            'file_path': file_path,
            'file_name': file_name,
            'frames': frames,
            'label': label
        }

        return sample

    def __len__(self):
        return len(self.video_paths)

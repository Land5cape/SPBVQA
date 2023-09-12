import argparse
import os
import time

KoNViD_1k_video_dir = 'E:/workplace/VQAdataset/KoNViD-1k'
KoNViD_1k_label_path = 'E:/workplace/VQAdataset/KoNViD-1k/KoNViD_1k_attributes.csv'
KoNViD_1k_split_path = './datas/KoNViD_1k_split.csv'
KoNViD_1k_MOS = [1.22, 4.64]

CVD2014_video_dir = 'E:/workplace/VQAdataset/CVD2014'
CVD2014_label_path = 'E:/workplace/VQAdataset/CVD2014/Realignment_MOS.csv'
CVD2014_split_path = './datas/CVD2014_split.csv'
CVD2014_MOS = [-6.50, 93.38]

LiveQ_video_dir = 'E:/workplace/VQAdataset/Live-Qualcomm'
LiveQ_label_path = 'E:/workplace/VQAdataset/Live-Qualcomm/qualcommSubjectiveData.csv'
LiveQ_split_path = './datas/LiveQ_split.csv'
LiveQ_MOS = [16.5621, 73.6428]

LiveV_video_dir = 'E:/workplace/VQAdataset/Live-VQC'
LiveV_label_path = 'E:/workplace/VQAdataset/Live-VQC/LiveVQCData.csv'
LiveV_split_path = './datas/LiveV_split.csv'
LiveV_MOS = [6.2237, 94.2865]

UGC_video_dir = 'E:/workplace/VQAdataset/YouTube-UGC'
UGC_label_path = 'E:/workplace/VQAdataset/YouTube-UGC/original_videos_MOS_for_YouTube_UGC_dataset.csv'
UGC_split_path = './datas/UGC_split.csv'
UGC_MOS = [1.242, 4.698]

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

INPUT_LENGTH = 8
INPUT_SIZE = 256
SEG_STRIDE = 1
CLIP_STRIDE = 8

RESIZE = False
RESIZE_SIZE = 512

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 3e-4
EPOCH_NUM = 30
BATCH_SIZE = 32  # 32

ds = 4
if ds == 1:
    video_dir = KoNViD_1k_video_dir
    label_path = KoNViD_1k_label_path
    split_path = KoNViD_1k_split_path
    mos = KoNViD_1k_MOS
    tail = '.mp4'
elif ds == 2:
    video_dir = CVD2014_video_dir
    label_path = CVD2014_label_path
    split_path = CVD2014_split_path
    mos = CVD2014_MOS
    tail = '.avi'
elif ds == 3:
    video_dir = LiveQ_video_dir
    label_path = LiveQ_label_path
    split_path = LiveQ_split_path
    mos = LiveQ_MOS
    tail = '.yuv'
elif ds == 4:
    video_dir = LiveV_video_dir
    label_path = LiveV_label_path
    split_path = LiveV_split_path
    mos = LiveV_MOS
    tail = '.mp4'
elif ds == 5:
    video_dir = UGC_video_dir
    label_path = UGC_label_path
    split_path = UGC_split_path
    mos = UGC_MOS
    tail = '.mp4'


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='train', type=str, help='train, retrain or predict')

    parser.add_argument('--video_dir', default=video_dir, type=str, help='Path to input videos')
    parser.add_argument('--video_type', default=tail, type=str, help='Type of videos')

    parser.add_argument('--score_file_path', default=label_path, type=str,
                        help='Path to input subjective score')
    parser.add_argument('--info_file_path', default=split_path, type=str,
                        help='Path to input info')
    parser.add_argument('--MOS_min', default=mos[0], type=float, help='MOS min range')
    parser.add_argument('--MOS_max', default=mos[1], type=float, help='MOS max range')

    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=WEIGHT_DECAY, type=float, help='L2 regularization')
    parser.add_argument('--epoch_num', default=EPOCH_NUM, type=int, help='epochs to train')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')

    start_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    parser.add_argument('--start_time', default=start_time, type=str,
                        help='start time of this process')
    parser.add_argument('--save_model', default='./model-save/' + str(ds) + '_' + start_time, type=str,
                        help='path to save the model')
    parser.add_argument('--save_checkpoint', default=True, type=bool, help='')
    parser.add_argument('--load_model', default='', type=str,
                        help='path to load checkpoint')
    parser.add_argument('--writer_t_dir', default='./runs/' + str(ds) + '_' + start_time + '_train', type=str,
                        help='batch size to train')
    parser.add_argument('--writer_v_dir', default='./runs/' + str(ds) + '_' + start_time + '_val', type=str,
                        help='batch size to train')

    args = parser.parse_args()

    if not os.path.isdir('./model-save/'):
        os.mkdir('./model-save/')
    if not os.path.isdir('./runs/'):
        os.mkdir('./runs/')

    return args


if __name__ == '__main__':
    args = parse_opts()
    print(args)

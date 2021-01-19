#!/usr/bin/env python3

# Copyright 2020 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script split data into training and validation set
#


import os
import torch
import numpy as np
import random
from pathlib import Path

def load_data(data_file):
    """ return dictionary { rec: data } """
    lines = [line.strip().split(None, 1) for line in open(data_file)]
    return {x[0]: x[1] for x in lines}


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='Data directory path')
parser.add_argument('train_data_dir', type=str,
                    help='Data directory path')
parser.add_argument('valid_data_dir', type=str,
                    help='Data directory path')
parser.add_argument('-nt', '--num_training_data', type=int, default=None,
                    help='Traininig data dir.')
parser.add_argument('-nv', '--num_validation_data', type=int, default=None,
                    help='Validation data dir.')
args = parser.parse_args()

data_dir = Path(args.data_dir)
train_data_dir = Path(args.train_data_dir)
train_data_dir.mkdir(parents=True, exist_ok=True)
valid_data_dir = Path(args.valid_data_dir)
valid_data_dir.mkdir(parents=True, exist_ok=True)


wav_scp = load_data(data_dir / 'wav.scp')
feats_scp = load_data(data_dir / 'feats.scp')
utt2num_frames = load_data(data_dir / 'utt2num_frames')
utt2spk = load_data(data_dir / 'utt2spk')


utt_list = list(utt2spk.keys())
len_list = len(utt_list)
idx_list = list(range(len_list))
random.shuffle(idx_list)

num_train = args.num_training_data
num_valid = args.num_validation_data
assert len_list >= num_train + num_valid, \
    'Number of all data ({}) is smaller than the number of subset data ({})'.format(len_list, num_train + num_valid)

train_list = [utt_list[i] for i in sorted(idx_list[:num_train])]
valid_list = [utt_list[i] for i in sorted(idx_list[num_train:num_train+num_valid])]


# Write training data
wav_scp_wf = open(train_data_dir / 'wav.scp', 'w')
feats_scp_wf = open(train_data_dir / 'feats.scp', 'w')
utt2num_frames_wf = open(train_data_dir / 'utt2num_frames', 'w')
utt2spk_wf = open(train_data_dir / 'utt2spk', 'w')

for utt in train_list:
    wav_scp_wf.write('{} {}\n'.format(utt,wav_scp[utt]))
    feats_scp_wf.write('{} {}\n'.format(utt,feats_scp[utt]))
    utt2num_frames_wf.write('{} {}\n'.format(utt,utt2num_frames[utt]))
    utt2spk_wf.write('{} {}\n'.format(utt,utt2spk[utt]))

wav_scp_wf.close()
feats_scp_wf.close()
utt2num_frames_wf.close()
utt2spk_wf.close()


# Write validation data
wav_scp_wf = open(valid_data_dir / 'wav.scp', 'w')
feats_scp_wf = open(valid_data_dir / 'feats.scp', 'w')
utt2num_frames_wf = open(valid_data_dir / 'utt2num_frames', 'w')
utt2spk_wf = open(valid_data_dir / 'utt2spk', 'w')

for utt in valid_list:
    wav_scp_wf.write('{} {}\n'.format(utt,wav_scp[utt]))
    feats_scp_wf.write('{} {}\n'.format(utt,feats_scp[utt]))
    utt2num_frames_wf.write('{} {}\n'.format(utt,utt2num_frames[utt]))
    utt2spk_wf.write('{} {}\n'.format(utt,utt2spk[utt]))

wav_scp_wf.close()
feats_scp_wf.close()
utt2num_frames_wf.close()
utt2spk_wf.close()
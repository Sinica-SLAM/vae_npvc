#!/usr/bin/env python3

# Copyright 2021 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script generate data list of VCTK corpus
# Split data with utt_id into test(1~25) / dev(26~50) / train(51~) set

import os
from pathlib import Path

ROOT_PATH = '/mnt/md0/user_roland/VCTK-Corpus/wav'
LIST_PATH = 'data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_root', type=str, default=ROOT_PATH,
                    help='Corpus path')
parser.add_argument('-l','--list_dir', type=str, default=LIST_PATH,
                    help='directory to output list')
parser.add_argument('-f','--fs', type=int, default=48000,
                    help='sampling rate')
args = parser.parse_args()

data_root = Path(args.data_root)
list_dir = Path(args.list_dir)

train_dir = list_dir / 'vctk_train'
train_dir.mkdir(parents=True, exist_ok=True)
wf_train_wavscp = open(str(train_dir/'wav.scp'),'w')
wf_train_utt2spk = open(str(train_dir/'utt2spk'),'w')
train_spk2utt = dict()

dev_dir = list_dir / 'vctk_dev'
dev_dir.mkdir(parents=True, exist_ok=True)
wf_dev_wavscp = open(str(dev_dir/'wav.scp'),'w')
wf_dev_utt2spk = open(str(dev_dir/'utt2spk'),'w')
dev_spk2utt = dict()

test_dir = list_dir / 'vctk_test'
test_dir.mkdir(parents=True, exist_ok=True)
wf_test_wavscp = open(str(test_dir/'wav.scp'),'w')
wf_test_utt2spk = open(str(test_dir/'utt2spk'),'w')
test_spk2utt = dict()

for speaker_dir in sorted(list(data_root.glob('*'))):
    speaker_name = speaker_dir.stem
    train_spk2utt[speaker_name] = list()
    dev_spk2utt[speaker_name] = list()
    test_spk2utt[speaker_name] = list()
    for data_file in sorted(list(speaker_dir.glob('*.wav'))):
        data_name = data_file.stem
        data_path = str(data_file.absolute())
        data_command = 'sox {} -c 1 -r {} -b 16 -t wav - |'.format(data_path, args.fs)
        data_num = int(data_name.split('_')[-1])
        if data_num <= 0:
            wf_test_wavscp.write('{} {}\n'.format(data_name,data_command))
            wf_test_utt2spk.write('{} {}\n'.format(data_name,speaker_name))
            test_spk2utt[speaker_name].append(data_name)
        elif data_num > 0 and data_num <= 50:
            wf_dev_wavscp.write('{} {}\n'.format(data_name,data_command))
            wf_dev_utt2spk.write('{} {}\n'.format(data_name,speaker_name))
            dev_spk2utt[speaker_name].append(data_name)
        else:
            wf_train_wavscp.write('{} {}\n'.format(data_name,data_command))
            wf_train_utt2spk.write('{} {}\n'.format(data_name,speaker_name))
            train_spk2utt[speaker_name].append(data_name)

wf_train_wavscp.close()
wf_train_utt2spk.close()
wf_dev_wavscp.close()
wf_dev_utt2spk.close()
wf_test_wavscp.close()
wf_test_utt2spk.close()

with open(str(train_dir/'spk2utt'),'w') as wf:
    for speaker_name, utt_names in train_spk2utt.items():
        if len(utt_names) > 0:
            wf.write('{} {}\n'.format(speaker_name,' '.join(utt_names)))

with open(str(dev_dir/'spk2utt'),'w') as wf:
    for speaker_name, utt_names in dev_spk2utt.items():
        if len(utt_names) > 0:
            wf.write('{} {}\n'.format(speaker_name,' '.join(utt_names)))

with open(str(test_dir/'spk2utt'),'w') as wf:
    for speaker_name, utt_names in test_spk2utt.items():
        if len(utt_names) > 0:
            wf.write('{} {}\n'.format(speaker_name,' '.join(utt_names)))
#!/usr/bin/env python3

# Copyright 2021 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script generate data list of VCC2020 corpus
# Split data with utt_id into dev(26~50) / train(51~) set

import os
from pathlib import Path

ROOT_PATH = '/mnt/md0/user_roland/vcc2020_training'
LIST_PATH = 'data'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_root', type=str, default=ROOT_PATH,
                    help='Corpus path')	
parser.add_argument('-l','--list_dir', type=str, default=LIST_PATH,
                    help='directory to output list')	
args = parser.parse_args()


speaker_list = ['SEF1','SEF2','SEM1','SEM2','TEF1','TEF2','TEM1','TEM2']

data_root = Path(args.data_root)
list_dir = Path(args.list_dir)

train_dir = list_dir / 'vcc2020_train'
train_dir.mkdir(parents=True, exist_ok=True)
wf_train_wavscp = open(str(train_dir/'wav.scp'),'w')
wf_train_utt2spk = open(str(train_dir/'utt2spk'),'w')
train_spk2utt = dict()

dev_dir = list_dir / 'vcc2020_dev'
dev_dir.mkdir(parents=True, exist_ok=True)
wf_dev_wavscp = open(str(dev_dir/'wav.scp'),'w')
wf_dev_utt2spk = open(str(dev_dir/'utt2spk'),'w')
dev_spk2utt = dict()

for speaker_name in sorted(speaker_list):
    train_spk2utt[speaker_name] = list()
    dev_spk2utt[speaker_name] = list()
    for data_file in sorted(list((data_root/speaker_name).glob('*.wav'))):
        data_num = int(data_file.stem[-2:])
        utt_name = '{}_{}'.format(speaker_name,data_file.stem)
        if data_num >= 51 and data_num <= 70:
            wf_dev_wavscp.write('{} {}\n'.format(utt_name,str(data_file.absolute())))
            wf_dev_utt2spk.write('{} {}\n'.format(utt_name,speaker_name))
            dev_spk2utt[speaker_name].append(utt_name)
        else:
            wf_train_wavscp.write('{} {}\n'.format(utt_name,str(data_file.absolute())))
            wf_train_utt2spk.write('{} {}\n'.format(utt_name,speaker_name))
            train_spk2utt[speaker_name].append(utt_name)

wf_train_wavscp.close()
wf_train_utt2spk.close()
wf_dev_wavscp.close()
wf_dev_utt2spk.close()

with open(str(dev_dir/'spk2utt'),'w') as wf:
    for speaker_name, utt_names in dev_spk2utt.items():
        if len(utt_names) > 0:
            wf.write('{} {}\n'.format(speaker_name,' '.join(utt_names)))

with open(str(train_dir/'spk2utt'),'w') as wf:
    for speaker_name, utt_names in train_spk2utt.items():
        if len(utt_names) > 0:
            wf.write('{} {}\n'.format(speaker_name,' '.join(utt_names)))




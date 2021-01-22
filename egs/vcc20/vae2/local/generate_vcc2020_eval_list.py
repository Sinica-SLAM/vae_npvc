#!/usr/bin/env python3

# Copyright 2021 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script generate data list of VCC2020 corpus
# Split data with utt_id into dev(26~50) / train(51~) set

import os
from pathlib import Path

ROOT_PATH = '/mnt/md0/user_roland/vcc2020_evaluation'
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

eval_dir = list_dir / 'vcc2020_test'
eval_dir.mkdir(parents=True, exist_ok=True)

wf_eval_wavscp = open(str(eval_dir/'wav.scp'),'w')
wf_eval_utt2spk = open(str(eval_dir/'utt2spk'),'w')
eval_spk2utt = dict()

for speaker_name in sorted(speaker_list):
    eval_spk2utt[speaker_name] = list()
    for data_file in sorted(list((data_root/speaker_name).glob('*.wav'))):
        utt_name = '{}_{}'.format(speaker_name,data_file.stem)
        wf_eval_wavscp.write('{} {}\n'.format(utt_name,str(data_file.absolute())))
        wf_eval_utt2spk.write('{} {}\n'.format(utt_name,speaker_name))
        eval_spk2utt[speaker_name].append(utt_name)

wf_eval_wavscp.close()
wf_eval_utt2spk.close()

with open(str(eval_dir/'spk2utt'),'w') as wf:
    for speaker_name, utt_names in eval_spk2utt.items():
        if len(utt_names) > 0:
            wf.write('{} {}\n'.format(speaker_name,' '.join(utt_names)))
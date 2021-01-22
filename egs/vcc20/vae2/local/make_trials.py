#!/usr/bin/env python3

# Copyright 2021 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script generate trials list of VCC2020 corpus
# 

import os
import numpy as np
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str,
                    help='directory to the list')
parser.add_argument('-s','--source_list', type=str, default=[], nargs='+',
                    help='Name of source speaker')
parser.add_argument('-t','--target_list', type=str, default=[], nargs='+',
                    help='Name of source speaker')
parser.add_argument('-n','--num_of_trials', type=int, default=20,
                    help='Number of trials')
parser.add_argument('-p','--parallel', action='store_true',
                    help='It\'s parallel data or not')
args = parser.parse_args()

data_dir = Path(args.data_dir)

source_list = [None] if len(args.source_list) == 0 else args.source_list
target_list = [None] if len(args.target_list) == 0 else args.target_list    

with open(data_dir / 'spk2utt','r') as rf:
    spk2utt = [line.rstrip().split() for line in rf.readlines()]

spk2utt = dict([[line[0],line[1:]] for line in spk2utt])
spk_list = spk2utt.keys()
spk_num = len(spk_list)

with open(data_dir / 'trials','w') as wf:
    n_trials = 0
    for source_name_ in source_list:
        for target_name_ in target_list:
            for n in range(args.num_of_trials):
                if source_name_ is None:
                    spk_idx = np.random.randint(0, spk_num)
                    source_name = spk_list[spk_idx]
                else:
                    source_name = source_name_

                if target_name_ is None:
                    spk_idx = np.random.randint(0, spk_num)
                    target_name = spk_list[spk_idx]
                else:
                    target_name = target_name_

                num_utts = len(spk2utt[source_name])
                idx = n if n < num_utts else n%num_utts
                source_utt = spk2utt[source_name][idx+10]

                if args.parallel and idx < len(spk2utt[target_name]):
                    target_utt = spk2utt[target_name][idx]
                    wf.write('{} {} {} {}\n'.format(source_utt, source_name, target_utt, target_name))
                    print('Trial {}: {} {} {} {}'.format(n_trials, source_utt, source_name, target_utt, target_name))
                else:
                    wf.write('{} {} {}\n'.format(source_utt, source_name, target_name))
                    print('Trial {}: {} {} {}'.format(n_trials, source_utt, source_name, target_name))

                n_trials += 1

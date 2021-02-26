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
parser.add_argument('-s','--source', type=str, default='',
                    help='Name of source speaker')
parser.add_argument('-t','--target', type=str, default='',
                    help='Name of target speaker')
parser.add_argument('-f','--format', type=str, default='S-T',
                    help='trials format. "S" for "Source", "T" for "Target"')
args = parser.parse_args()

data_dir = Path(args.data_dir)

source = None if len(args.source) == 0 else args.source
target = None if len(args.target) == 0 else args.target
assert target is not None

spk_format = args.format.split('-')

with open(data_dir / 'utt2spk','r') as rf:
    utt2spk = [line.rstrip().split() for line in rf.readlines()]

with open(data_dir / 'trials','w') as wf:
    n_trials = 0
    for utt,spk in utt2spk:
        if source and spk != source:
            continue

        trial = [utt]
        for spk_kind in spk_format:
            if spk_kind.upper() in ['S','SOURCE']:
                trial.append(spk)
            elif spk_kind.upper() in ['T','TARGET']:
                trial.append(target)
        trial = ' '.join(trial)

        wf.write('{}\n'.format(trial))
        print('Trial {}: {}'.format(n_trials, trial))

        n_trials += 1

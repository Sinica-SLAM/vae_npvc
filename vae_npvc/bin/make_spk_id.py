#!/usr/bin/env python3

# Copyright 2019 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script masks all the name of the dataset
#

import os
import numpy as np
from pathlib import Path

def load_data(data_file):
    """ return dictionary { rec: data } """
    lines = [line.strip().split(None, 1) for line in open(data_file)]
    return {x[0]: x[1] for x in lines}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    help='input data dir')
parser.add_argument('--spk2spk_id', default=None,
                    help='spk2spk_id data file')
args = parser.parse_args()

data_dir = Path(args.data_dir)

if args.spk2spk_id is None:
    spk2utt = load_data(data_dir / 'spk2utt')
    print('Get {} speakers from {}'.format(len(spk2spk_id.keys()), str(data_dir / 'spk2utt') ))

    spk2spk_id = {spk:'{:06d}'.format(i) for i, spk in enumerate(spk2utt.keys())}
    del spk2utt

    with open(data_dir / 'spk2spk_id','w') as wf:
        for spk, spkid in spk2spk_id.items():
            content = '{} {}'.format(spk, spkid)
            wf.write(content+'\n')

else:
    spk2spk_id = load_data(args.spk2spk_id)
    print('Get {} speakers from {}'.format(len(spk2spk_id.keys()), str(args.spk2spk_id) ))


utt2spk = load_data(data_dir / 'utt2spk')
print('Get {} utterances from {}'.format(len(utt2spk.keys()), str(data_dir / 'utt2spk') ))
with open(data_dir / 'utt2spk_id','w') as wf:
    for utt, spk in utt2spk.items():
        if spk not in spk2spk_id.keys():
            continue
        content = '{} {}'.format(utt, spk2spk_id[spk])
        wf.write(content+'\n')
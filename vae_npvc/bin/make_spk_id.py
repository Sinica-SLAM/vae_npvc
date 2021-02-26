#!/usr/bin/env python3

# Copyright 2019 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script masks all the name of the dataset
#

import os
import numpy as np
from pathlib import Path
from shutil import copyfile

def load_data(data_file):
    """ return dictionary { rec: data } """
    lines = [line.strip().split(None, 1) for line in open(data_file)]
    return {x[0]: x[1] for x in lines}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    help='input data dir')
parser.add_argument('--spk2spk_id', type=str, default='',
                    help='spk2spk_id data file')
parser.add_argument('--write_utt2spk_id', type=str, default='true',
                    help='write utt2spk_id or not')
args = parser.parse_args()

data_dir = Path(args.data_dir)
 
if args.spk2spk_id == '':
    if (data_dir / 'spk2spk_id').exists():
        print('{} exists, use it.'.format(str(data_dir / 'spk2spk_id')))
        spk2spk_id = load_data(data_dir / 'spk2spk_id')
        print('Get {} speakers from {}'.format(len(spk2spk_id.keys()), str(data_dir / 'spk2spk_id') ))

    else:
        print('{} does not exist, generate it from spk2utt.'.format(str(data_dir / 'spk2spk_id')))
        assert (data_dir / 'spk2utt').exists(), '{} does not exist'.format(str(data_dir / 'spk2utt'))

        spk2utt = load_data(data_dir / 'spk2utt')
        print('Get {} speakers from {}'.format(len(spk2utt.keys()), str(data_dir / 'spk2utt') ))

        spk2spk_id = {spk:'{:06d}'.format(i) for i, spk in enumerate(spk2utt.keys())}
        del spk2utt

        with open(data_dir / 'spk2spk_id','w') as wf:
            for spk, spkid in spk2spk_id.items():
                content = '{} {}'.format(spk, spkid)
                wf.write(content+'\n')
else:
    assert Path(args.spk2spk_id).exists(), 'No such file {}'.format(args.spk2spk_id)

    if (data_dir / 'spk2spk_id').exists() and str(data_dir / 'spk2spk_id') != args.spk2spk_id:
        backup_dir = data_dir / '.backup'
        backup_dir.mkdir(parents=True, exist_ok=True)
        print('Backup {} to {}'.format(str(data_dir / 'spk2spk_id'), str(backup_dir) ))
        os.rename(str(data_dir / 'spk2spk_id'), str(backup_dir / 'spk2spk_id') )
    copyfile(args.spk2spk_id, str(data_dir / 'spk2spk_id'))

    spk2spk_id = load_data(args.spk2spk_id)
    print('Get {} speakers from {}'.format(len(spk2spk_id.keys()), str(args.spk2spk_id) ))

if args.write_utt2spk_id.lower() == 'true':
    utt2spk = load_data(data_dir / 'utt2spk')
    print('Generate utt2spk_id, get {} utterances from {}'.format(len(utt2spk.keys()), str(data_dir / 'utt2spk') ))
    with open(data_dir / 'utt2spk_id','w') as wf:
        for utt, spk in utt2spk.items():
            if spk not in spk2spk_id.keys():
                print('Warning: speaker "{}" does not in the speaker id list'.format(spk))
                continue
            content = '{} {}'.format(utt, spk2spk_id[spk])
            wf.write(content+'\n')

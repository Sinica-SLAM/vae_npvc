#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('phone_file', type=str,
                    help='TEXT file')
parser.add_argument('text_file', type=str,
                    help='Output text')
parser.add_argument('dict_dir', type=str,
                    help='Directory path for phonetic symbol file')
args = parser.parse_args()

dict_dir = Path(args.dict_dir)
dict_dir.mkdir(parents=True, exist_ok=True)

text = [line.rstrip().split() for line in open(args.phone_file)]
exclude = ['<end>']

if not (dict_dir / 'phonetic_symbols').exists():
  
    phonetic_symbols = []
    for content in text:
        for _sym in content[1:]:
            if _sym in exclude:
                continue

            _sym = _sym.split('<')[-1].split('>')[0]
            if _sym not in phonetic_symbols:
                phonetic_symbols.append(_sym)


    with open(dict_dir / 'phonetic_symbols', 'w') as wf:
        wf.write('<unk>\n')
        for _sym in phonetic_symbols:
            wf.write('<{}>\n'.format(_sym))
    with open(dict_dir / 'phonetic_dictionary', 'w') as wf:
        wf.write('<unk> 1\n')
        for _id, _sym in enumerate(phonetic_symbols):
            wf.write('<{}> {}\n'.format(_sym, _id+2))
else:
    phonetic_symbols = [
        line.rstrip().split('<')[-1].split('>')[0]
        for line in open(dict_dir / 'phonetic_symbols')
    ]

with open(args.text_file, 'w') as wf:
    for content in text:
        uid = content[0]
        wf.write('{} '.format(uid))

        content = content[1:]
        for _sym in content:        
            if _sym in exclude:
                continue

            _sym = _sym.split('<')[-1].split('>')[0]
            if _sym in phonetic_symbols:
                wf.write('<{}>'.format(_sym))

        wf.write('\n')


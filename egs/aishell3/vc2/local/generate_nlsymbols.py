import os
import sys
from pathlib import Path

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("data_dir", type=str,
                    help="data dir")
parser.add_argument("-n", "--num_symbol", type=int, default=512,
                    help="Number of non-language symbols")
args = parser.parse_args()

data_dir = Path(args.data_dir)
num_symbol = args.num_symbol

data_dir.mkdir(parents=True, exist_ok=True)

sym_path = data_dir / 'symbols'
dict_path = data_dir / 'dictionary'

with open(sym_path,'w') as wf:
    wf.write('<unk>\n')
    for i in range(num_symbol):
        wf.write('<{}>\n'.format(i))

with open(dict_path,'w') as wf:
    wf.write('<unk> 1\n')
    for i in range(num_symbol):
        wf.write('<{}> {}\n'.format(i,i+2))

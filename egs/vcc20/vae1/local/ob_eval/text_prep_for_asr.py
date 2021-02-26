#!/usr/bin/env python3

import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('text_in', type=str,
                    help='Text input')
parser.add_argument('scp_file', type=str,
                    help='Needed SCP file')
parser.add_argument('text_out', type=str,
                    help='Text input')
args = parser.parse_args()

id2trs = dict([l.rstrip().split(None,1) for l in open(args.text_in)])
utt_list = [l.rstrip().split(None,1)[0] for l in open(args.scp_file)]

with open(args.text_out,'w') as wf:
	for utt in utt_list:
		wf.write('{} {}\n'.format(utt, id2trs[utt.split('E')[-1]]))
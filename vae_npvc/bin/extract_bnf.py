#!/usr/bin/env python3

# Copyright 2020 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script masks all the name of the dataset
#

import os
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import math

import torch
import torch.nn.functional as F

from kaldiio import ReadHelper, WriteHelper

MAX_WAV_VALUE = 32768.0


def extract_bnf( args):
    model_path = args.model_path
    bnf_kind = args.bnf_kind
    output_txt = True if args.output_txt.lower() in ['true'] else False

    rspecifier = args.rspecifier
    wspecifier = args.wspecifier

    config = yaml.safe_load(open(args.config))

    model_type = config.get('model_type', 'vae_npvc.model.vqvae')
    module = import_module(model_type, package=None)
    model = getattr(module, 'Model')(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    model.cuda().eval()
    
    if output_txt and bnf_kind in ['id','csid']:
        bnf_writer = open(wspecifier,'w')
    else:
        bnf_writer = WriteHelper(bnf_writer, compression_method=1)
        output_txt = False

    for utt, feat in ReadHelper(rspecifier):
        # Load source features
        
        feat_in = torch.from_numpy(np.array(feat))
        feat_in = feat_in.float().cuda().t().unsqueeze(0)
   
        with torch.no_grad():
            bnf_out = model.encode(feat_in).unsqueeze(-1)
            
        # Save converted feats
        if bnf_kind == 'id':
            bnf_out = bnf_out.view(-1).cpu().numpy()
        elif bnf_kind == 'csid':
            bnf_out = bnf_out.view(-1).unique_consecutive().cpu().numpy()
        elif bnf_kind == 'token':
            bnf_out = bnf_out.t().cpu().numpy()

        if output_txt:
            bnf_out = bnf_out.reshape(-1)
            bnf_out = ''.join(['<{}>'.format(bnf) for bnf in bnf_out])
            bnf_writer.write('{} {}\n'.format(utt,bnf_out))
        else:
            bnf_writer.write(utt, bnf_out)

        print('Extracting BNF {} of {}.'.format( bnf_kind, utt),end=' '*30+'\r')

    print('Finished Extracting BNF {}.'.format( bnf_kind),end=' '*len(utt)+'\n')
    bnf_writer.close()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/utt2spks.yaml',
                        help='YAML file for configuration')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to checkpoint with model')
    parser.add_argument('--bnf_kind', type=str, default=None,
                        help='Bottleneck feature kinds.')
    parser.add_argument('--output_txt', type=str, default='true')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Using gpu #')
    parser.add_argument('rspecifier', type=str,
                        help='Input specifier')
    parser.add_argument('wspecifier', type=str,
                        help='Output specifier')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    extract_bnf(args)




#!/usr/bin/env python3

# Copyright 2019 Yu-Huai Peng
# Licensed under the MIT license.
#
# This script masks all the name of the dataset
#

import os
import sys

import time
import logging
from pathlib import Path
from importlib import import_module
import shutil

import numpy as np

import torch


def decode(args): 
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint
    decode_dir = Path(args.decode_dir)

    config = yaml.safe_load(open(args.config))
    config.update({'use_gpu':False if args.gpu[0] == 'c' else True})

    decoder_type         = config.get('decoder_type', 'vae_npvc.decoder.basic:Decoder').split(':')
    seed                 = config.get('seed', 777)

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initial trainer
    decoder_module = import_module(decoder_type[0], package=None)
    decoder_name = 'Decoder' if len(decoder_type) < 2 else decoder_type[1]
    decoder = getattr( decoder_module, decoder_name)(config)

    # Load checkpoint if the path is given 
    assert checkpoint_path is not None
    _ = decoder.load_checkpoint(checkpoint_path)
    
    # Prepare logger
    logger = logging.getLogger()
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=str(output_dir/'decode.log'))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    # logger.info(decoder.get_model_info())
    logger.info("Decoding dataset: {}".format(decode_dir))
    logger.info("Decoding model: {}".format(checkpoint_path))

    # ================ MAIN TRAINNIG LOOP! ===================
    
    logger.info("Start decoding...")

    decoder.decode(decode_dir, output_dir)

    print('Finished')
        

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/utt2spks.yaml',
                        help='YAML file for configuration')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for checkpoint output')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint path to keep training')
    parser.add_argument('--decode-dir', type=str, default=None,
                        help='Evaluation data dir.')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='Using gpu id; If get "cpu" or "c", use cpu')
    args = parser.parse_args()

    if args.gpu[0] != 'c':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    decode(args)

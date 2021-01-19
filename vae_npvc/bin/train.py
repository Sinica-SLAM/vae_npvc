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

import numpy as np

import torch
from torch.utils.data import DataLoader


def train(args): 
    # Setup
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint
    train_dir = args.train_dir
    valid_dir = args.valid_dir

    config = yaml.safe_load(open(args.config))

    trainer_type         = config.get('trainer_type', 'vae_npvc.trainer.basic')
    dataset_type         = config.get('dataset_type', 'vae_npvc.dataset.utt2mel_spkid')
    max_iter             = config.get('max_iter', 100000)
    iters_per_checkpoint = config.get('iters_per_checkpoint', 10000)
    iters_per_log        = config.get('iters_per_log', 1000)
    num_jobs             = config.get('num_jobs', 8)
    prefetch_factor      = config.get('prefetch_factor', 2)
    seed                 = config.get('seed', 777)

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initial trainer
    trainer_module = import_module(trainer_type, package=None)
    trainer = getattr( trainer_module, 'Trainer')(config)

    # Load checkpoint if the path is given 
    iteration = 1
    if checkpoint_path is not None:
        iteration = trainer.load_checkpoint(checkpoint_path)
        iteration += 1  # next iteration is iteration + 1

    # Load training data
    dataset_module = import_module(dataset_type, package=None)
    Dataset = getattr( dataset_module, 'Dataset')
    try:
        collate_fn = getattr( dataset_module, 'collate')
    except:
        collate_fn = None
    batch_size = config.get('train_batch_size', config.get('batch_size', 32))
    train_set = Dataset(train_dir, config)
    train_loader = DataLoader(
        train_set, 
        num_workers=num_jobs, shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
    )

    # Load validation data
    try:
        batch_size = config.get('valid_batch_size', config.get('batch_size', 1))
        valid_set = Dataset(valid_dir, config, valid=True)
        valid_loader = DataLoader(
            valid_set, 
            num_workers=num_jobs, shuffle=False,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
        )
    except:
        valid_set, valid_loader = [], None

    # Get shared output_directory ready
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=str(output_dir/'train.log'))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info(trainer.get_model_info())
    logger.info("Output directory: {}".format(output_dir))
    logger.info("Training utterances: {}".format(len(train_set)))
    logger.info("Validation utterances: {}".format(len(valid_set)))

    # ================ MAIN TRAINNIG LOOP! ===================
    
    logger.info("Start traininig...")

    train_log = dict()
    best_model = None
    best_loss = np.inf
    check_loss = 'loss'

    while iteration <= max_iter:
        for i, batch in enumerate(train_loader):    

            iteration, loss_detail = trainer.train_step(batch, iteration=iteration)

            # Keep Loss detail
            for key,val in loss_detail.items():
                if key not in train_log.keys():
                    train_log[key] = list()
                train_log[key].append(val)

            # Show log per M iterations
            if iteration % iters_per_log == 0 and len(train_log.keys()) > 0:
                mseg = 'Iter {}:'.format( iteration)
                for key,val in train_log.items():
                    mseg += '  {}: {:.6f}'.format(key,np.mean(val))
                logger.info(mseg)
                train_log = dict()

            # Save model per N iterations
            if iteration % iters_per_checkpoint == 0:
                output_name = "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()), iteration)
                checkpoint_path =  output_dir / output_name
                trainer.save_checkpoint( checkpoint_path)
                logger.info("Saved state dict. to {}".format(checkpoint_path))

            # Validation per N iterations
            if iteration % iters_per_checkpoint == 0 and valid_loader is not None:
                loss_detail = trainer.valid(valid_loader)
                mseg = 'Valid {}:'.format( iteration)
                for key,val in loss_detail.items():
                    mseg += '  {}: {:.6f}'.format(key,np.mean(val))
                logger.info(mseg)

            # Check if finished
            if iteration > max_iter:
                break

    logger.info("Finished")
        

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/utt2spks.yaml',
                        help='YAML file for configuration')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for checkpoint output')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint path to keep training')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Traininig data dir.')
    parser.add_argument('--valid_dir', type=str, default=None,
                        help='Validation data dir.')

    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='Using gpu #')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    train(args)

import os
import numpy as np
import torch
import torch.nn.functional as F
import random
from pathlib import Path

# from kaldi.util.io import read_matrix
from kaldiio import load_mat

def load_dict_data(data_file):
    """ return dictionary { rec: data } """
    lines = [line.strip().split(None, 1) for line in open(data_file)]
    return {x[0]: x[1] for x in lines}

def load_list_data(data_file):
    """ return list [ rec, data ] """
    return [line.strip().split() for line in open(data_file)]


class Dataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_dir, config, valid=False):
        if valid:
            crop_length       = config.get('crop_length', 256)
            self.crop_length  = config.get('valid_crop_length', crop_length)
        else:
            crop_length       = config.get('crop_length', 256)
            self.crop_length  = config.get('train_crop_length', crop_length)
        self.valid = valid

        data_dir = Path(data_dir)
        self.feats_scp = load_dict_data(data_dir / 'feats.scp')
        self.utt2num_frames = load_dict_data(data_dir / 'utt2num_frames')
        self.utt2spks = load_list_data(data_dir / 'utt2spk_id')

        self.num_data = len(self.utt2spks)
 
    def __getitem__(self, index):
        '''
        Output:
            feat: Size(Time, Dim)
            feat_len: Size(1)
            target_spks: Size(MaxSpkNum)
            spks_len: Size(1)
        '''        
        # Read audio
        utt, spk = self.utt2spks[index]
        feat_length = int(self.utt2num_frames[utt])

        if feat_length <= self.crop_length:
            feat_start = 0
            feat_end = feat_length
        else:
            max_feat_start = feat_length - self.crop_length
            feat_start = random.randint(0, max_feat_start) if not self.valid else 0
            feat_end = feat_start + self.crop_length

        feat_scp = str(self.feats_scp[utt])
        feat_scp += '[{}:{}]'.format(feat_start,feat_end-1)
        # feat = read_matrix(feat_scp).numpy()        # Size(num_frames, num_dim)
        feat = np.array(load_mat(feat_scp))         # Size(num_frames, num_dim)
        feat = torch.from_numpy(feat.T).float()     # Size(num_dim, num_frames)

        if feat_length < self.crop_length:
            diff_length = self.crop_length - feat_length
            feat = F.pad(feat, (0, diff_length)).data

        spk = torch.tensor([int(spk)]).long()

        return feat, spk


    def __len__(self):
        return self.num_data


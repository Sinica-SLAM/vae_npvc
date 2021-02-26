import torch
import numpy as np
from pathlib import Path
from importlib import import_module
from kaldiio import load_mat, WriteHelper

import logging
logger = logging.getLogger()

class Decoder(object):
    def __init__(self, config):
        model_type     = config.get('model_type', 'utt2spks.model.utt2spks:Model').split(':')

        if config.get('use_gpu', True):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        
        module = import_module(model_type[0], package=None)
        model_name = 'Model' if len(model_type) < 2 else model_type[1]
        model = getattr(module, model_name)(config)
        self.model = model.to(self.device)   
        self.model.eval()


    def decode_step(self, feat, spk):
        # Try decoding. If fail, use cpu decoding.
        with torch.no_grad():
            try:
                outputs = self.model.infer((feat, spk))
            except:
                self.model = self.model.cpu()
                outputs = self.model.infer((feat.cpu(), spk.cpu()))
                self.model = self.model.to(self.device)
        return outputs


    def decode(self, decode_dir, output_dir, compress=True):
        decode_dir = Path(decode_dir)
        output_dir = str(output_dir)
        for file in ['trials','feats.scp']:
            file_ = decode_dir / file
            if not file_.is_file():
                logger.info('No such file {}'.format(str(file_)))
                assert False

        trials = [line.strip().split(None,1) for line in open(str(decode_dir / 'trials'))]
        feats_scp = dict([line.strip().split(None,1) for line in open(str(decode_dir / 'feats.scp'))])
        if (decode_dir / 'spk2spk_id').exists():
            spk2spk_id = dict([line.strip().split(None,1) for line in open(str(decode_dir / 'spk2spk_id'))])
        else:
            spk2spk_id = None

        wspecifier = 'ark,scp:{0}/feats.ark,{0}/feats.scp'.format(str(output_dir))
        compress_opt = {'compression_method': 1}
        with WriteHelper(wspecifier, **compress_opt) as wf:
            for i, (utt, target) in enumerate(trials):
                logger.info(f'Decode {i}: {utt} to {target}')

                feat = np.array(load_mat(feats_scp[utt]))
                feat = torch.from_numpy(feat).float().to(self.device)
                feat = feat.t().unsqueeze(0)

                if spk2spk_id:
                    target = [int(spk2spk_id[t]) for t in target.split()]
                else:
                    target = [int(t) for t in target.split()]
                target = torch.tensor(target).long().to(self.device)
                target = target.view(1,-1)

                feat_decode = self.decode_step(feat, target)
                feat_decode = feat_decode[0].t().detach().cpu().numpy()

                wf[utt] = feat_decode



    def get_model_info(self):
        return self.model

    def load_checkpoint(self, checkpoint_file):
        checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
        self.model.load_state_dict(checkpoint_data['model'])
        return checkpoint_data['iteration']


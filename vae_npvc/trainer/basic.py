import torch

from importlib import import_module

from .radam import RAdam 

import math
LOG_2PI = math.log( 2.0 * math.pi)

class Trainer(object):
    def __init__(self, config):
        learning_rate  = config.get('learning_rate', 1e-4)
        model_type     = config.get('model_type', 'vae_npvc.model.vqvae')
        optim_type     = config.get('optim_type', 'Adam')
        learning_rate  = config.get('learning_rate', 1)
        max_grad_norm  = config.get('max_grad_norm', 5)
        lr_scheduler   = config.get('lr_scheduler', None)
        lr_param       = config.get('lr_param', {
                                        'step_size': 100000,
                                        'gamma': 0.5,
                                        'last_epoch': -1
                                    })

        module = import_module(model_type, package=None)
        MODEL = getattr(module, 'Model')
        model = MODEL(config)

        self.model = model.cuda()
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        if optim_type.upper() == 'RADAM':
            self.optimizer = RAdam( self.model.parameters(), 
                                    lr=learning_rate,
                                    betas=(0.5,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.parameters(),
                                               lr=learning_rate,
                                               betas=(0.5,0.999),
                                               weight_decay=0.0)

        if lr_scheduler is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                optimizer=self.optimizer, **lr_param
                            )
        else:
            self.scheduler = None


        self.iteration = 0
        self.model.train()


    def train_step(self, input, iteration=None):
        assert self.model.training
        self.model.zero_grad()

        input = [x.cuda() for x in input]
        output, loss, loss_detail = self.model(input)

        loss_detail['Total'] = loss.item()

        loss.backward()
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if iteration is None:
            self.iteration = iteration + 1
        else:
            self.iteration += 1

        return self.iteration, loss_detail


    def valid(self, data_loader):
        loss_detail = dict()
        for i, batch in enumerate(data_loader):
            step_detail = self.valid_step(batch)
            for key, val in step_detail.items():
                if key not in loss_detail.keys():
                    loss_detail[key] = list()
                loss_detail[key].append(val)

        return loss_detail


    def valid_step(self, input):
        self.model.eval()

        with torch.no_grad():
            input = [x.cuda() for x in input]
            loss_detail = self.model(input)

        self.model.train()

        return loss_detail


    def get_model_info(self):
        return self.model

    def save_checkpoint(self, checkpoint_file):
        torch.save( {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_file)
        print("Saved state dict. to {}".format(checkpoint_file))

    def load_checkpoint(self, checkpoint_file):
        checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
        self.model.load_state_dict(checkpoint_data['model'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer'])
        return checkpoint_data['iteration']

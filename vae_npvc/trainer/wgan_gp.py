import os
import math
import torch
import torch.nn.functional as F

from .radam import RAdam

from .losses import gradient_penalty_loss


class Trainer(object):
    def __init__(self, config):
        self._gamma        = config.get('gamma', 1)
        self._gp_weight    = config.get('gp_weight', 1)
        self.pre_iter      = config.get('pre_iter', 1000)
        self.gen_param     = config.get('generator_param', {
                                'per_iteration': 1,
                                'optim_type': 'RAdam',
                                'learning_rate': 1e-4,
                                'max_grad_norm': 10,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })
        self.disc_param    = config.get("discriminator_param", {
                                'per_iteration': 1,            
                                'optim_type': 'RAdam',
                                'learning_rate': 5e-5,
                                'max_grad_norm': 1,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })


        checkpoint_path    = config.get('checkpoint_path', '')


        # Initial Generator and Discriminator
        module = import_module('utt2spks.model.{}'.format(model_type), package=None)
        MODEL = getattr(module, 'Model')
        self.model_G = MODEL(config['Generator'])
        DISCRIM = getattr(module, 'Discriminator')
        self.model_D = DISCRIM(config['Discriminator'])

        print(self.model_G)
        print(self.model_D)

        # Initial Optimizer
        self.optimizer_G = RAdam( self.model_G.parameters(), 
                                  lr=self.gen_param['learning_rate'],
                                  betas=(0.5,0.999),
                                  weight_decay=0.0)

        self.optimizer_D = RAdam( self.model_D.parameters(), 
                                  lr=self.disc_param['learning_rate'],
                                  betas=(0.5,0.999),
                                  weight_decay=0.0)

        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
                                    optimizer=self.optimizer_G,
                                    **self.gen_param['lr_scheduler']
                            )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
                                    optimizer=self.optimizer_D,
                                    **self.disc_param['lr_scheduler']
                            )

        if os.path.exists(checkpoint_path):
            self.iteration = self.load_checkpoint(checkpoint_path)
        else:
            self.iteration = 0

        self.model_G.cuda().train()
        self.model_D.cuda().train()

    def step(self, input, iteration=None):
        if iteration is None:
            iteration = self.iteration

        assert self.model_G.training 
        assert self.model_D.training

        x_batch, y_batch = input
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        loss_detail_G = dict()
        loss_detail_D = dict()

        ##########################
        # Phase 1: Train the VAE #
        ##########################
        if iteration <= self.pre_iter:
            x_output, loss, loss_detail_G = self.model_G((x_batch, y_batch))

            self.model_G.zero_grad()
            loss.backward()
            if self.gen_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_G.parameters(),
                    self.gen_param['max_grad_norm'])
            self.optimizer_G.step()
            self.scheduler_G.step()

        ####################################
        # Phase 2: Train the discriminator #
        ####################################
        if iteration > self.pre_iter and iteration % self.disc_param['per_iteration'] == 0:
            # Train the Discriminator
            with torch.no_grad():
                x_output, _, _  = self.model_G((x_batch, y_batch))

            logit_real = - self.model_D(x_batch).mean()
            logit_fake = self.model_D(x_output).mean()
            gp_loss =  gradient_penalty_loss(x_batch, x_output, self.model_D)

            disc_loss = logit_real + logit_fake
            loss = disc_loss + self._gp_weight * gp_loss

            loss_detail_D['DISC loss'] = disc_loss.item()
            loss_detail_D['gradient_penalty'] = gp_loss.item()

            self.model_G.zero_grad()
            self.model_D.zero_grad()
            loss.backward()
            if self.disc_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_D.parameters(),
                    self.disc_param['max_grad_norm'])
            self.optimizer_D.step()
            self.scheduler_D.step()

        ################################
        # Phase 2: Train the generator #
        ################################
        if iteration > self.pre_iter and iteration % self.gen_param['per_iteration'] == 0:
            # Train the Generator
            x_output, loss, loss_detail_G  = self.model_G((x_batch, y_batch))

            adv_loss = - self.model_D(x_output)
            loss += self._gamma * adv_loss

            loss_detail_G['Total'] = loss.item()
            loss_detail_G['ADV loss'] = adv_loss.item()
            
            self.model_G.zero_grad()
            self.model_D.zero_grad()
            loss.backward()
            if self.gen_param['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model_G.parameters(),
                    self.gen_param['max_grad_norm'])
            self.optimizer_G.step()
            self.scheduler_G.step()

        # Get loss detail
        loss_detail = dict()
        for key, val in loss_detail_G.items():
            loss_detail[key] = val
        for key, val in loss_detail_D.items():
            loss_detail[key] = val

        self.iteration = iteration + 1

        return self.iteration, loss_detail


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model_G.state_dict(),
                'discriminator': self.model_D.state_dict(),
                'optimizer_G': self.optimizer_G.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint_data.keys():
            self.model_G.load_state_dict(checkpoint_data['model'])
        if 'discriminator' in checkpoint_data.keys():
            self.model_D.load_state_dict(checkpoint_data['discriminator'])
        if 'optimizer_G' in checkpoint_data.keys():
            self.optimizer_G.load_state_dict(checkpoint_data['optimizer_G'])
        if 'optimizer_D' in checkpoint_data.keys():
            self.optimizer_D.load_state_dict(checkpoint_data['optimizer_D'])
        self.scheduler_G.last_epoch = checkpoint_data['iteration']
        self.scheduler_D.last_epoch = checkpoint_data['iteration'] - self.pre_iter
        return checkpoint_data['iteration']



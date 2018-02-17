from __future__ import print_function
from argparse import ArgumentParser
import logging

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import transforms

from ignite.trainer import Trainer
from ignite.evaluator import Evaluator
from ignite.engine import Events
from ignite.handlers.evaluate import Evaluate
import numpy as np

from architectures import image2image

import os
import util
import itertools
from skimage.io import imsave
from skimage.transform import rescale


def save_vis_image(A_real, atob, atob_btoa, B_real, btoa, btoa_atob, out_file, scale_factor=1.):
    """
    Saves an image to disk where each row in the image corresponds to
      [A_real, atob, atob_btoa, B_real, btoa, btoa_atob] and the number
      of rows are determined the first axis (batch axis) of these
      tensors.
    """
    outs_np = [A_real, atob, atob_btoa, B_real, btoa, btoa_atob]
    # determine # of channels
    n_channels = outs_np[0].shape[1]
    shp = outs_np[0].shape[-1]
    # possible that A_real.bs != B_real.bs
    bs = np.min([outs_np[0].shape[0], outs_np[3].shape[0]])
    grid = np.zeros((shp*bs, shp*6, 3))
    for j in range(bs):
        for i in range(6):
            n_channels = outs_np[i][j].shape[1]
            img_to_write = util.convert_to_rgb(outs_np[i][j], is_grayscale=True if n_channels==1 else False)
            grid[j*shp:(j+1)*shp,i*shp:(i+1)*shp,:] = img_to_write
    imsave(arr=rescale(grid, scale=scale_factor), fname=out_file)

def get_log_all_losses_handler(logger, prefix=""):
    def log_all_losses(evaluator):
        # TODO: make this nicer??
        loss_bufs = [0.] * 6
        for i in range(len(evaluator.history)):
            for j in range(len(loss_bufs)):
                loss_bufs[j] += evaluator.history[i][j]
        for j in range(len(loss_bufs)):
            loss_bufs[j] /= len(evaluator.history)
        logger("[%s] atob_gen_loss=%f, btoa_gen_loss=%f, cycle_aba_loss=%f, cycle_bab_loss=%f, d_a_loss=%f, d_b_loss=%f" %
               (prefix, loss_bufs[0], loss_bufs[1], loss_bufs[2], loss_bufs[3], loss_bufs[4], loss_bufs[5]))
    return log_all_losses


def run():

    logger = print
    gen_params = {'input_nc':3, 'ngf':64, 'output_nc':3, 'norm':'instance', 'which_model_netG':'resnet_9blocks'}
    disc_params = {'input_nc':3, 'ndf':64, 'n_layers_D':3, 'norm':'instance', 'which_model_netD':'n_layers'}
    g_atob = image2image.define_G(**gen_params)
    g_btoa = image2image.define_G(**gen_params)
    d_a = image2image.define_D(**disc_params)
    d_b = image2image.define_D(**disc_params)
    print("g_atob")
    print(g_atob)
    print("# params: %i" % util.count_params(g_atob))
    print("d_a")
    print(d_a)
    print("# params: %i" % util.count_params(d_a))    
    
    train_loader, val_loader = get_kaggle_folder_iterators(batch_size=1, labels_file=os.environ["DR_FOLDERS"] + "/trainLabels.csv")
    #import pdb
    #pdb.set_trace()

    opt_args = { 'lr':0.0002, 'betas':(0.5, 0.999) }
    optim_g = Adam( itertools.chain(g_atob.parameters(), g_btoa.parameters()), **opt_args)
    optim_d_a = Adam( d_a.parameters(), **opt_args)
    optim_d_b = Adam( d_b.parameters(), **opt_args)

    mse_loss = torch.nn.MSELoss()
    lamb = 10.
    use_cuda = True

    num_pool = 50
    fake_A_pool = util.ImagePool(num_pool)
    fake_B_pool = util.ImagePool(num_pool)

    if use_cuda:
        g_atob.cuda()
        g_btoa.cuda()
        d_a.cuda()
        d_b.cuda()

    def compute_g_losses(A_real, atob, atob_btoa, B_real, btoa, btoa_atob):
        """Return all the losses related to generation"""
        d_a_fake = d_a(btoa)
        d_b_fake = d_b(atob)
        ones_da = torch.ones(d_a_fake.size())
        ones_db = torch.ones(d_b_fake.size())
        if use_cuda:
            ones_da, ones_db = ones_da.cuda(), ones_db.cuda()
        ones_da, ones_db = Variable(ones_da), Variable(ones_db)
        btoa_gen_loss = mse_loss(d_a_fake, ones_da)
        atob_gen_loss = mse_loss(d_b_fake, ones_db)
        cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
        cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
        g_tot_loss = atob_gen_loss + btoa_gen_loss + lamb*cycle_aba + lamb*cycle_bab
        return atob_gen_loss, btoa_gen_loss, cycle_aba, cycle_bab, g_tot_loss

    def compute_d_losses(A_real, atob, atob_btoa, B_real, btoa, btoa_atob):
        """Return all losses related to discriminator"""
        d_a_real = d_a(A_real)
        d_b_real = d_b(B_real)
        A_fake = fake_A_pool.query(btoa)
        B_fake = fake_B_pool.query(atob)
        d_a_fake = d_a(A_fake)
        d_b_fake = d_b(B_fake)
        ones_da_real, zeros_da_fake = torch.ones(d_a_real.size()), torch.zeros(d_a_fake.size())
        ones_db_real, zeros_db_fake = torch.ones(d_b_real.size()), torch.zeros(d_b_fake.size())
        if use_cuda:
            ones_da_real, zeros_da_fake, ones_db_real, zeros_db_fake = \
                        ones_da_real.cuda(), zeros_da_fake.cuda(), ones_db_real.cuda(), zeros_db_fake.cuda()
        ones_da_real, zeros_da_fake, ones_db_real, zeros_db_fake = \
                        Variable(ones_da_real), Variable(zeros_da_fake), Variable(ones_db_real), Variable(zeros_db_fake)
        d_a_loss = (mse_loss(d_a_real, ones_da_real) + mse_loss(d_a_fake, zeros_da_fake)) * 0.5
        d_b_loss = (mse_loss(d_b_real, ones_db_real) + mse_loss(d_b_fake, zeros_db_fake)) * 0.5
        return d_a_loss, d_b_loss

    def training_update_function(results_dir=None, save_images_every=1000, images_scale_factor=1.):
        """
        Parameters
        ----------
        results_dir: directory used to store visualisations and other
          useful information.
        save_images_every: save the images in `results_dir` every this many
          iterations (calls) of the inner function
        images_scale_factor: scale factor for saved visualisations (good
          for saving disk space when the dumped images are very big)
        """
        num_iters = 0
        def _fn(batch):
            nonlocal num_iters
            g_atob.train()
            g_btoa.train()
            d_a.train()
            d_b.train()
            A_real, B_real = batch
            A_real, B_real = A_real.float(), B_real.float()
            if use_cuda:
                A_real, B_real = A_real.cuda(), B_real.cuda()
            A_real, B_real = Variable(A_real), Variable(B_real)
            atob = g_atob(A_real)
            btoa = g_btoa(B_real)
            atob_btoa = g_btoa(atob)
            btoa_atob = g_atob(btoa)
            # update generator
            # TODO: log these other losses
            atob_gen_loss, btoa_gen_loss, cycle_aba, cycle_bab, g_tot_loss = \
                    compute_g_losses(A_real, atob, atob_btoa, B_real, btoa, btoa_atob)
            optim_g.zero_grad()
            g_tot_loss.backward()
            optim_g.step()
            # update discriminator
            d_a_loss, d_b_loss = compute_d_losses(A_real, atob, atob_btoa, B_real, btoa, btoa_atob)
            optim_d_a.zero_grad()
            d_a_loss.backward()
            optim_d_a.step()
            optim_d_b.zero_grad()
            d_b_loss.backward()
            optim_d_b.step()
            num_iters += 1
            # dump images
            if num_iters % save_images_every == 0 and results_dir is not None:
                outs = [ x.data.cpu().numpy() for x in [A_real, atob, atob_btoa, B_real, btoa, btoa_atob] ]
                save_vis_image(*outs, out_file="%s/%06d_train.png" % (results_dir, num_iters), scale_factor=images_scale_factor)
            return atob_gen_loss.data[0], btoa_gen_loss.data[0], cycle_aba.data[0], cycle_bab.data[0], d_a_loss.data[0], d_b_loss.data[0]
        return _fn

    def validation_function(results_dir=None, save_images_every=1000, images_scale_factor=1.):
        num_iters = 0
        def _fn(batch):
            nonlocal num_iters
            g_atob.eval()
            g_btoa.eval()
            d_a.eval()
            d_b.eval()
            A_real, B_real = batch
            A_real, B_real = A_real.float(), B_real.float()
            if use_cuda:
                A_real, B_real = A_real.cuda(), B_real.cuda()
            A_real, B_real = Variable(A_real), Variable(B_real)
            atob = g_atob(A_real)
            btoa = g_btoa(B_real)
            atob_btoa = g_btoa(atob)
            btoa_atob = g_atob(btoa)
            atob_gen_loss, btoa_gen_loss, cycle_aba, cycle_bab, g_tot_loss = \
                    compute_g_losses(A_real, atob, atob_btoa, B_real, btoa, btoa_atob)
            d_a_loss, d_b_loss = compute_d_losses(A_real, atob, atob_btoa, B_real, btoa, btoa_atob)
            num_iters += 1
            # dump images
            if num_iters % save_images_every == 0 and results_dir is not None:
                outs = [ x.data.cpu().numpy() for x in [A_real, atob, atob_btoa, B_real, btoa, btoa_atob] ]
                save_vis_image(*outs, out_file="%s/%06d_valid.png" % (results_dir, num_iters), scale_factor=images_scale_factor)
            return atob_gen_loss.data[0], btoa_gen_loss.data[0], cycle_aba.data[0], cycle_bab.data[0], d_a_loss.data[0], d_b_loss.data[0]
        return _fn

    trainer = Trainer(
        training_update_function(results_dir="tmp", save_images_every=5000)
    )
    evaluator = Evaluator(
        validation_function(results_dir="tmp", save_images_every=5000)
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, Evaluate(evaluator, val_loader, epoch_interval=1))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, get_log_all_losses_handler(logger, 'train'))
    
    # evaluator event handlers
    evaluator.add_event_handler(Events.COMPLETED, get_log_all_losses_handler(logger, 'valid'))

    # kick everything off
    epochs = 10
    trainer.run(train_loader, max_epochs=epochs)

if __name__ == '__main__':
    run()

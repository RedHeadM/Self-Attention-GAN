
import datetime
import os
import time

import numpy as np

import torch
import torch.nn as nn
from sagan_models import Discriminator, Generator
from torch.autograd import Variable
from torch.nn import functional as F
from torchtcn.utils.dataset import DoubleViewPairDataset
from torchvision.utils import save_image
from utils import *
from collections import OrderedDict
try:
    import visdom
    vis = visdom.Visdom()
    vis.env = 'vae_dcgan'
except (ImportError, AttributeError):
    vis = None
    print("visdom not used")


class Trainer(object):
    def __init__(self, data_loader, config):

        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.cam_view_z = 15+30+10
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        self.vae_rec_path = os.path.join(config.sample_path, "vae_rec")
        os.makedirs(self.vae_rec_path, exist_ok=True)  # TODO
        print('vae_rec_path: {}'.format(self.vae_rec_path))
        self.model_save_path = os.path.join(config.model_save_path, self.version)

        self.num_pixels = self.imsize * 2 * 3

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        fixed_z = None

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()

        num_views = 2
        key_views = ["frames views {}".format(i) for i in range(num_views)]

        lable_keys_cam_view_info = []  # list with keys for view 0 and view 1
        for view_i in range(num_views):
            lable_keys_cam_view_info.append(["cam_pitch_view_{}".format(view_i),
                                             "cam_yaw_view_{}".format(view_i),
                                             "cam_distance_view_{}".format(view_i)])

        mapping_cam_info_lable = OrderedDict()
        mapping_cam_info_one_hot = OrderedDict()
        # create a different mapping for echt setting
        n_classes = []
        for cam_info_view in lable_keys_cam_view_info:
            for cam_inf in cam_info_view:
                if "pitch" in cam_inf:
                    min_val, max_val = -50, -35.
                    n_bins = 15
                elif "yaw" in cam_inf:
                    min_val, max_val = -60., 210.
                    n_bins = 30
                elif "distance" in cam_inf:
                    min_val, max_val = 0.7, 1.
                    n_bins = 10

                to_l, to_hot_l = create_lable_func(min_val, max_val, n_bins)
                mapping_cam_info_lable[cam_inf] = to_l
                mapping_cam_info_one_hot[cam_inf] = to_hot_l
                if "view_0" in cam_inf:
                    n_classes.append(n_bins)
        print('n_classes: {}'.format(n_classes))

        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            if isinstance(self.data_loader.dataset, DoubleViewPairDataset):
                data = next(data_iter)
                # real_images = torch.cat([data[key_views[0]], data[key_views[1]]])
                #  for now only view 0
                real_images = data[key_views[0]]
                label_c = OrderedDict()
                label_c_hot_in = OrderedDict()
                for key_l, lable_func in mapping_cam_info_lable.items():
                    # contin cam values to labels
                    label_c[key_l] = torch.tensor(lable_func(data[key_l])).cuda()
                    label_c_hot_in[key_l] = torch.tensor(
                        mapping_cam_info_one_hot[key_l](data[key_l]), dtype=torch.float32).cuda()
                d_one_hot = [label_c_hot_in[l] for l in lable_keys_cam_view_info[0]]
            else:
                real_images, _ = next(data_iter)
            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            d_out_real, dr1, dr2 = self.D(real_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z=torch.randn(real_images.size(0), self.z_dim).cuda()
            z=torch.cat([*d_one_hot,z],dim=1) # add view info
            if  fixed_z is None:
                fixed_z=tensor2var(torch.randn(self.batch_size, self.z_dim))
                fixed_z=torch.cat([*d_one_hot,fixed_z],dim=1)# add view info
            z = tensor2var(z)
            fake_images, gf1, gf2 = self.G(z)
            d_out_fake, df1, df2 = self.D(fake_images)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha)
                                        * fake_images.data, requires_grad=True)
                out, _, _ = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train VAE================== #
            encoded = self.G.encoder(real_images)
            mu = encoded[0]
            logvar = encoded[1]
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            # KLD = torch.sum(KLD_element).mul_(-0.5)

            sampled = self.G.encoder.sampler(encoded)
            z=torch.cat([*d_one_hot,sampled],dim=1)# add view info
            z = tensor2var(z) # TODO ok?
            fake_images, _, _ = self.G(z)
            MSEerr = self.MSECriterion(fake_images, real_images)
            # Reconstruction loss is pixel wise cross-entropy
            # a = fake_images.view(-1 , self.num_pixels)
            # b = real_images.view(-1, self.num_pixels)
            # MSEerr = F.binary_cross_entropy(denorm(a),
                                            # denorm(b))
            # MSEerr = F.binary_cross_entropy(fake_images.view(-1, self.num_pixels),
            #                                 real_images.view(-1, self.num_pixels))
            rec = fake_images
            VAEerr = MSEerr
            # VAEerr = KLD + MSEerr
            self.reset_grad()
            # VAEerr.backward()
            # self.g_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            # z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            # fake_images, _, _ = self.G(z)

            # Compute loss with fake images
            g_out_fake, _, _ = self.D(fake_images)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            # g_loss_fake.backward()
            loss = g_loss_fake+VAEerr*self.num_pixels
            loss.backward()

            self.g_optimizer.step()

            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f},vae {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step, d_loss_real,
                             self.G.attn1.gamma.mean(), self.G.attn2.gamma.mean(), VAEerr))
                if vis is not None:
                    kw_update_vis = None

                    if self.d_plot is not None:
                        kw_update_vis = 'append'
                        # kw_update_vis["update"] = 'append'
                    self.d_plot = vis.line(np.array([d_loss_real.detach().cpu().numpy()]), X=np.array(
                        [step]), win=self.d_plot, update=kw_update_vis, opts=dict(
                        title="d_loss_real",
                        xlabel='Timestep',
                        ylabel='loss'
                    ))
                    self.d_plot_fake = vis.line(np.array([d_loss_fake.detach().cpu().numpy()]), X=np.array(
                        [step]), win=self.d_plot_fake, update=kw_update_vis, opts=dict(
                        title="d_loss_fake",
                        xlabel='Timestep',
                        ylabel='loss'
                    ))
                    self.d_plot_vae = vis.line(np.array([VAEerr.detach().cpu().numpy()]), X=np.array(
                        [step]), win=self.d_plot_vae, update=kw_update_vis, opts=dict(
                        title="VAEerr",
                        xlabel='Timestep',
                        ylabel='loss'
                    ))

            # Sample images
            if True or  (step + 1) % self.sample_step == 0:
                fake_images, _, _ = self.G(fixed_z)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))
                n = 8
                imgs = denorm(torch.cat([real_images.data[:n], rec.data[:n]]))
                title = '{}_var_rec'.format(step + 1)
                save_image(imgs,
                           os.path.join(self.vae_rec_path, title+".png"), nrow=n)
                if vis is not None:
                    self.rec_win = vis.images(imgs, win=self.rec_win,
                                              opts=dict(title=title, width=64*n, height=64*2),)

            if (step+1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):
        self.rec_win = None
        self.d_plot = None
        self.d_plot_fake = None
        self.d_plot_vae = None
        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.cam_view_z, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.MSECriterion = nn.MSELoss()
        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))


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
from sklearn.utils import shuffle
try:
    import visdom
    vis = visdom.Visdom()
    vis.env = 'vae_dcgan'
    vis.close()#close all
except (ImportError):
    vis = None
    print("visdom not used")


class Trainer(object):
    def __init__(self, loader, config, data_loader_val=None):

        # Data loader
        data_loader, data_loader_val = loader
        self.data_loader = data_loader
        self.data_loader_val = data_loader_val

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.cam_view_z = (20+40+10)*2+5
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
        self.model_save_path = os.path.join(
            config.model_save_path, self.version)

        self.num_pixels = self.imsize * 2 * 3

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):
        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        # Using itertools.cycle has an important drawback, in that it does not shuffle the data after each iteration:
        # WARNING  itertools.cycle  does not shuffle the data after each iteratio
        # Data iterator
        data_iter = iter(cycle(self.data_loader))
        self.loader_val_iter = iter(cycle(self.data_loader_val))
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
                    n_bins = 20
                elif "yaw" in cam_inf:
                    min_val, max_val = -60., 210.
                    n_bins = 40
                elif "distance" in cam_inf:
                    min_val, max_val = 0.7, 1.
                    n_bins = 10

                to_l, to_hot_l = create_lable_func(min_val, max_val, n_bins)
                mapping_cam_info_lable[cam_inf] = to_l
                mapping_cam_info_one_hot[cam_inf] = to_hot_l
                if "view_0" in cam_inf:
                    n_classes.append(n_bins)
        print('cam view one hot infputs {}'.format(n_classes))
        task_progess_bins =5
        _,task_progress_hot_func=create_lable_func(0,115,n_bins=task_progess_bins,clip=True)

        assert sum(n_classes) * 2+task_progess_bins == self.cam_view_z

        def changing_factor(start,end, steps):
            for i in range(steps):
                yield i/(steps/(end-start))+start
        cycle_factor_gen=changing_factor(0.5,1.,self.total_step)
        triplet_factor_gen=changing_factor(0.1,1.,self.total_step)
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            if isinstance(self.data_loader.dataset, DoubleViewPairDataset):
                data = next(data_iter)
                key_views, lable_keys_cam_view_info = shuffle(key_views, lable_keys_cam_view_info)
                # real_images = torch.cat([data[key_views[0]], data[key_views[1]]])
                #  for now only view 0
                real_images = data[key_views[0]]
                real_images_view1 = data[key_views[1]]
                label_c = OrderedDict()
                label_c_hot_in = OrderedDict()
                for key_l, lable_func in mapping_cam_info_lable.items():
                    # contin cam values to labels
                    label_c[key_l] = torch.tensor(
                        lable_func(data[key_l])).cuda()
                    label_c_hot_in[key_l] = torch.tensor(
                        mapping_cam_info_one_hot[key_l](data[key_l]), dtype=torch.float32).cuda()
                d_one_hot_view0 = [label_c_hot_in[l]
                                   for l in lable_keys_cam_view_info[0]]
                d_one_hot_view1 = [label_c_hot_in[l] for l in lable_keys_cam_view_info[1]]
                d_task_progress= torch.tensor(task_progress_hot_func(data['frame index']),dtype=torch.float32).cuda()
            else:
                real_images, _ = next(data_iter)
            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            real_images_view1 = tensor2var(real_images_view1)
            d_out_real, dr1, dr2 = self.D(real_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            encoded = self.G.encoder(real_images)
            sampled = self.G.encoder.sampler(encoded)
            z=torch.randn(real_images.size(0), self.z_dim).cuda()
            z = torch.cat([*d_one_hot_view0, *d_one_hot_view1,d_task_progress, sampled],
                          dim=1)  # add view info from to
            if fixed_z is None:
                fixed_z = tensor2var(
                    torch.cat([*d_one_hot_view0, *d_one_hot_view1,d_task_progress, sampled], dim=1))  # add view info
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
                                           grad_outputs=torch.ones(
                                               out.size()).cuda(),
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
            mu_0 = encoded[0]
            logvar = encoded[1]
            KLD_element = mu_0.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element).mul_(-0.5)
            # save_image(denorm(real_images[::2]), os.path.join(self.sample_path, "ancor.png"))
            # save_image(denorm(real_images[1::2]), os.path.join(self.sample_path, "neg.png"))
            # save_image(denorm(real_images_view1[::2]), os.path.join(self.sample_path, "pos.png"))


            sampled = self.G.encoder.sampler(encoded)
            z = torch.cat([*d_one_hot_view0, *d_one_hot_view1,d_task_progress, sampled], dim=1)  # add view info 0
            z = tensor2var(z)
            fake_images_0, _, _ = self.G(z)
            MSEerr = self.MSECriterion(fake_images_0, real_images_view1)
            rec = fake_images_0
            VAEerr = MSEerr +KLD
            # encode the fake view and recon loss to view1
            encoded = self.G.encoder(fake_images_0)
            mu_1 = encoded[0]
            logvar = encoded[1]
            KLD_element = mu_1.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element).mul_(-0.5)
            sampled = self.G.encoder.sampler(encoded)
            z = torch.cat([*d_one_hot_view1, *d_one_hot_view0,d_task_progress, sampled], dim=1)  # add view info 1
            z = tensor2var(z)
            fake_images_view1, _, _ = self.G(z)
            rec_fake = fake_images_view1
            MSEerr = self.MSECriterion(fake_images_view1, real_images)
            VAEerr += (KLD+MSEerr) *next(cycle_factor_gen)  # (KLD + MSEerr)  # *0.5
            triplet_loss = self.triplet_loss(
                anchor=mu_0[::2], positive=mu_0[1::2], negative=mu_1[::2])
            # ================== Train G and gumbel ================== #
            # Create random noise
            # z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            # fake_images, _, _ = self.G(z)

            # Compute loss with fake images
            # fake_images = torch.cat([fake_images_0, fake_images_view1]) # rm triplets
            fake_images = torch.cat([fake_images_0[::2], fake_images_view1[::2]]) # rm triplets
            g_out_fake, _, _ = self.D(fake_images)  # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            VAEerr*=self.num_pixels
            triplet_loss*=self.num_pixels
            loss = g_loss_fake*4.+VAEerr+triplet_loss*next(triplet_factor_gen)
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
                    self.d_plot_triplet_loss = vis.line(np.array([triplet_loss.detach().cpu().numpy()]), X=np.array(
                        [step]), win=self.d_plot_triplet_loss, update=kw_update_vis, opts=dict(
                        title="triplet_loss",
                        xlabel='Timestep',
                        ylabel='loss'
                    ))


            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images, _, _ = self.G(fixed_z)
                fake_images = denorm(fake_images)
                save_image(fake_images.data,
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))
                n = 8
                imgs = denorm(torch.cat([real_images.data[:n], rec.data[:n]]))
                imgs_rec_fake = denorm(torch.cat([real_images_view1.data[:n], rec_fake.data[:n]]))
                title = '{}_var_rec_real'.format(step + 1)
                title_rec_fake = '{}_var_rec_fake'.format(step + 1)
                title_fixed = '{}_fixed'.format(step + 1)
                save_image(imgs,
                           os.path.join(self.vae_rec_path, title+".png"), nrow=n)
                distance_pos, product_pos, distance_neg, product_neg = self._get_view_pair_distances()

                print("distance_pos {:.3}, neg {:.3},dot pos {:.3} neg {:.3}".format(
                    distance_pos, distance_neg, product_pos, product_neg))
                if vis is not None:
                    self.rec_win = vis.images(imgs, win=self.rec_win,
                                              opts=dict(title=title, width=64*n, height=64*2),)
                    self.rec_fake_win = vis.images(imgs_rec_fake, win=self.rec_fake_win,
                                                   opts=dict(title=title_rec_fake, width=64*n, height=64*2),)
                    self.fixed_win = vis.images(fake_images, win=self.fixed_win,
                                                opts=dict(title=title_fixed, width=64*n, height=64*4),)

                    kw_update_vis = None
                    if self.d_plot_distance_pos is not None:
                        kw_update_vis = 'append'
                    self.d_plot_distance_pos = vis.line(np.array([distance_pos]), X=np.array(
                        [step]), win=self.d_plot_distance_pos, update=kw_update_vis, opts=dict(
                        title="distance pos",
                        xlabel='Timestep',
                        ylabel='dist'
                    ))
                    self.d_plot_distance_neg = vis.line(np.array([distance_neg]), X=np.array(
                        [step]), win=self.d_plot_distance_neg, update=kw_update_vis, opts=dict(
                        title="distance neg",
                        xlabel='Timestep',
                        ylabel='dist'
                    ))
            if (step+1) % model_save_step == 0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def _get_view_pair_distances(self):
        def encode(x):
            encoded = self.G.encoder(x)
            mu = encoded[0]
            return mu
        # dot product are mean free
        key_views = ["frames views {}".format(i) for i in range(2)]
        sample_batched = next(self.loader_val_iter)
        anchor_emb = encode(sample_batched[key_views[0]].cuda())
        positive_emb = encode(sample_batched[key_views[1]].cuda())
        distance_pos = np.linalg.norm(
            anchor_emb.data.cpu().numpy() - positive_emb.data.cpu().numpy(), axis=1).mean()
        dots = []
        for e1, e2 in zip(anchor_emb.data.cpu().numpy(), positive_emb.data.cpu().numpy()):
            dots.append(np.dot(e1-e1.mean(), e2-e2.mean()))
        product_pos = np.mean(dots)

        n = len(anchor_emb)
        emb_pos = anchor_emb.data.cpu().numpy()
        emb_neg = positive_emb.data.cpu().numpy()
        cnt, distance_neg, product_neg = 0., 0., 0.
        for i in range(n):
            for j in range(n):
                if j != i:
                    d_negative = np.linalg.norm(
                        emb_pos[i] - emb_neg[j])
                    distance_neg += d_negative
                    product_neg += np.dot(emb_pos[i]-emb_pos[i].mean(),
                                          emb_neg[j]-emb_neg[j].mean())
                    cnt += 1
        distance_neg /= cnt
        product_neg /= cnt
        # distance_pos = np.asscalar(distance_pos)
        # product_pos = np.asscalar(product_pos)
        # distance_neg = np.asscalar(distance_neg)
        # product_neg = np.asscalar(product_neg)
        return distance_pos, product_pos, distance_neg, product_neg

    def build_model(self):
        self.rec_win = None
        self.rec_fake_win = None
        self.fixed_win = None
        self.d_plot = None
        self.d_plot_fake = None
        self.d_plot_vae = None
        self.d_plot_triplet_loss = None
        self.d_plot_distance_neg = None
        self.d_plot_distance_pos = None
        self.G = Generator(self.batch_size, self.imsize, self.z_dim,
                           self.cam_view_z, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
        self.MSECriterion = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss()
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
        save_image(denorm(real_images), os.path.join(
            self.sample_path, 'real.png'))

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(
            self.sample_path, 'real.png'))

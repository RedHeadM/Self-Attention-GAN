import torch
import torchvision.datasets as dsets
from torchvision import transforms
from torchtcn.utils.dataset import (DoubleViewPairDataset,)
from torchtcn.utils.sampler import ViewPairSequenceSampler


class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_tcn(self):
        transformer_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.imsize),
            transforms.ToTensor(),
            # normalize
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        sampler = None
        shuffle = True
        # only one view pair in batch
        # sim_frames = 5
        transformed_dataset_train = DoubleViewPairDataset(vid_dir=self.path,
                                                          number_views=2,
                                                          add_camera_info=True,
                                                          # std_similar_frame_margin_distribution=sim_frames,
                                                          transform_frames=transformer_train)
        transformed_dataset_val = DoubleViewPairDataset(vid_dir=self.path,
                                                        number_views=2,
                                                        # add_camera_info=True,
                                                        # std_similar_frame_margin_distribution=sim_frames,
                                                        transform_frames=transformer_train)
        sampler_val = ViewPairSequenceSampler(dataset=transformed_dataset_val,
                                              examples_per_sequence=self.batch,
                                              batch_size=self.batch)
        sampler_train = ViewPairSequenceSampler(dataset=transformed_dataset_val,
                                                examples_per_sequence=2,
                                                batch_size=self.batch)

        return transformed_dataset_train, sampler_train, transformed_dataset_val, sampler_val

    def loader(self):
        dataset_val, sampler_train, sampler_val, loader_val = None, None, None, None
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'tcn':
            dataset, sampler_train, dataset_val, sampler_val = self.load_tcn()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             # shuffle=self.shuf,
                                             sampler=sampler_train,
                                             num_workers=2,
                                             drop_last=True)
        if dataset_val is not None:
            loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                                     batch_size=self.batch,
                                                     shuffle=False,
                                                     sampler=sampler_val,
                                                     num_workers=2,
                                                     drop_last=True)

        return loader, loader_val

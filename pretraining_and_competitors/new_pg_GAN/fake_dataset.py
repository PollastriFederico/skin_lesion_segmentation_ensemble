from __future__ import print_function

import time
from my_code.new_pg_GAN.model import Generator

import torch.nn.functional as F
import torch.utils.data as data
import torch

'''
STATS

dermoscopic_wmasks
mean: tensor([0.3526, 0.0989, 0.0079]) | std: tensor([0.3490, 0.3380, 0.3431])

training_2017
mean: tensor([0.3359, 0.1133, 0.0276]) | std: tensor([0.3324, 0.3247, 0.3351])

validation_2017
mean: tensor([0.3853, 0.1210, 0.0207]) | std: tensor([0.1801, 0.1849, 0.2133])

test_2017
mean: tensor([0.4255, 0.1002, 0.0177]) | std: tensor([0.2022, 0.2189, 0.2370])

'''


class Fake(data.Dataset):
    """ FAKE Dataset. """

    def __init__(self, net_name='pgGAN', ckpt_name='training_2018', size=(512, 512), segmentation_transform=None,
                 transform=None, target_transform=None):
        start_time = time.time()
        self.segmentation_transform = segmentation_transform
        self.transform = transform
        self.target_transform = target_transform
        self.ckpt_name = ckpt_name
        self.size = size
        self.max_res = 6
        self.nch = 16
        self.nc = 4
        self.dl = 2000
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.netG = Generator(max_res=self.max_res, nch=self.nch, nc=self.nc, bn=False, ws=True, pn=True)
        self.netG = Generator(max_res=self.max_res, nch=self.nch, nc=self.nc, bn=False, ws=True, pn=True).to(self.DEVICE)
        self.netG.load_state_dict(torch.load(self.ckpt_name).state_dict())

        print("Time: " + str(time.time() - start_time))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, ground)
        """

        # z = torch.randn(1, 16 * 32, 1, 1)
        z = torch.randn(1, 16 * 32, 1, 1, device=self.DEVICE)

        with torch.no_grad():
            fake_image = self.netG(z, self.max_res)

            if self.size is not None:
                fake_image = F.interpolate(fake_image, self.size)

            fake_image = torch.squeeze(fake_image, 0)
            image = fake_image[0:3, :, :].clamp(-1, 1)
            ground = torch.unsqueeze(fake_image[-1, :, :], 0).clamp(0, 1)


            image = torch.add(image, 1)
            image = torch.div(image, 2)


        if self.segmentation_transform is not None:
            image, ground = self.segmentation_transform(image.cpu(), ground.cpu())
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            ground = self.target_transform(ground)

        return image, ground

    def __len__(self):
        return self.dl


import argparse
import time
import numpy as np
import os

import torch
from torch import nn
import torchvision.transforms as standard_transforms
from torchvision.utils import save_image

from torch.backends import cudnn
from torch.utils.data import DataLoader

from UNet import UNet as vanillaUnet
from Unet_noskips import UNet as Unet_noskips
from tiramisu import FCDenseNet67 as tiramisu
# from Deeplab.SplitDecoder import DeepLab
from Deeplab.decoder import DeepLab
from segnet import SegNet as segnet
from true_segnet import SegNet as true_segnet
from utils import jaccard
import glob

import my_code.new_pg_GAN.fake_dataset as fake_isic
import my_code.isic.isic_dataset as isic
import my_code.isic.transforms as transforms
from my_code.new_pg_GAN.progressBar import printProgressBar

cudnn.benchmark = True

pgsegm_root = '/homes/my_d/segm/'

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, default='isic')
parser.add_argument('--network', required=True, default='Unet')
parser.add_argument('--split', default='training_2017', help='dataset split')
parser.add_argument('--data', type=str, default='/homes/my_d/data/ISIC_dataset/Task_1/',
                    help='directory containing the data')
parser.add_argument('--outd', default=pgsegm_root + 'Results', help='directory to save results')
parser.add_argument('--outf', default=pgsegm_root + 'Images', help='folder to save synthetic images')
parser.add_argument('--outl', default=pgsegm_root + 'Losses', help='folder to save Losses')
parser.add_argument('--outm', default=pgsegm_root + 'Models', help='folder to save models')
parser.add_argument('--loadEpoch', type=int, default=0, help='load pretrained models')

parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=32,
                    help='list of batch sizes during the training')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the height / width of the input image to network')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='number of data loading workers')
parser.add_argument('--thresh', type=float, default=0.5, help='number of data loading workers')
parser.add_argument('--epochs', type=int, default=301, help='number of data loading workers')

parser.add_argument('--n_iter', type=int, default=50, help='number of epochs to train before changing the progress')
parser.add_argument('--savemodel', type=int, default=10, help='number of epochs between saving models')
parser.add_argument('--savemaxsize', action='store_true',
                    help='save sample images at max resolution instead of real resolution')
parser.add_argument('--ckpt_name', default='')
parser.add_argument('--load_ckpt_name', default=None)
parser.add_argument('--SRV', action='store_true', help='is training on remote server')

opt = parser.parse_args()
print(opt)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_RES = 6


class SegmentationNet:
    def __init__(self, num_epochs=301, l_r=1e-10, size=256, batch_size=32, n_workers=8, thresh=0.5, num_classes=3,
                 write_flag=False, net_name=None, ckpt_name=''):
        # Hyper-parameters
        self.num_epochs = num_epochs
        self.learning_rate = l_r
        self.size = size
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.thresh = thresh
        self.write_flag = write_flag
        self.num_classes = num_classes
        self.net_name = net_name

        self.ckpt_name = ckpt_name

        self.outf = os.path.join(pgsegm_root, opt.network, opt.split, opt.outf)
        self.outm = os.path.join(pgsegm_root, opt.network, opt.split, opt.outm)

        if opt.network == 'Unet':
            self.n = vanillaUnet(num_classes)
        elif opt.network == 'Unet_noskips':
            self.n = Unet_noskips(num_classes)
        elif opt.network == 'segnet':
            self.n = segnet(num_classes)
        elif opt.network == 'true_segnet':
            self.n = true_segnet(num_classes)
        elif opt.network == 'tiramisu':
            self.n = tiramisu(num_classes)
        elif opt.network == 'DeepLab':
            self.n = DeepLab(num_classes, pretrained=False)
        else:
            print("ERROR in network name")

        # allow data parallel
        if torch.cuda.device_count() > 1:
            self.n = nn.DataParallel(self.n)
        # put the model on DEVICE AFTER allowing data parallel
        self.n.to(DEVICE)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.n.parameters()), lr=self.learning_rate)

        if opt.dataset == 'isic':
            split_list = glob.glob('/homes/my_d/data/ISIC_dataset/Task_3/images/*.jpg')
            self.dataset = isic.ISIC(root=opt.data,  # split_list=training_set,
                                     split_list=split_list,
                                     load=False, size=(self.size, self.size),
                                     segmentation_transform=transforms.Compose([
                                         transforms.Resize((self.size, self.size)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomApply(
                                             [transforms.Resize((self.size, self.size)),
                                              transforms.ColorJitter(0.3, 0.3, 0.2, 0.01),
                                              transforms.RandomAffine(degrees=0, shear=5),
                                              transforms.RandomRotation(180),
                                              transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                              transforms.RandomAffine(degrees=0, scale=(0.95, 1.25))],
                                             p=0.4),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351)),
                                     ]),
                                     )

            self.val_dataset = isic.ISIC(root=opt.data,  # split_list=training_set,
                                         split_name='validation_2017',
                                         load=opt.SRV, size=(self.size, self.size),
                                         transform=standard_transforms.Compose([
                                             standard_transforms.Resize((self.size, self.size)),
                                             standard_transforms.ToTensor(),
                                             standard_transforms.Normalize((0.3359, 0.1133, 0.0276),
                                                                           (0.3324, 0.3247, 0.3351)),
                                         ]),
                                         target_transform=standard_transforms.Compose([
                                             standard_transforms.Resize((self.size, self.size)),
                                             standard_transforms.ToTensor()
                                         ])
                                         )

            self.test_dataset = isic.ISIC(root=opt.data,  # split_list=training_set,
                                          split_name='test_2017',
                                          load=False,
                                          transform=standard_transforms.Compose([
                                              standard_transforms.Resize((self.size, self.size)),
                                              standard_transforms.ToTensor(),
                                              standard_transforms.Normalize((0.3359, 0.1133, 0.0276),
                                                                            (0.3324, 0.3247, 0.3351)),
                                          ]),
                                          target_transform=standard_transforms.Compose([
                                              standard_transforms.ToTensor()
                                          ])
                                          )


        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.n_workers,
                                      drop_last=True,
                                      pin_memory=True)

        self.eval_data_loader = DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.n_workers,
                                           drop_last=False,
                                           pin_memory=True)

        self.test_data_loader = DataLoader(self.test_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=self.n_workers,
                                           drop_last=False,
                                           pin_memory=True)

        if not os.path.exists(self.outf):
            os.makedirs(self.outf)
        if not os.path.exists(self.outm):
            os.makedirs(self.outm)

        self.total = len(self.data_loader)
        print(len(self.data_loader))
        print(len(self.eval_data_loader))

    def train(self, epochs=None):
        if epochs:
            self.num_epochs = epochs
        sigm = nn.Sigmoid()
        for epoch in range(self.num_epochs):
            self.n.train()
            losses = []
            start_time = time.time()
            for i, (x, _) in enumerate(self.data_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # compute output
                x = x.to(DEVICE)
                target = x
                output = self.n(x)
                output = sigm(output)
                loss = self.criterion(output, target)
                losses.append(loss.item())
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if not opt.SRV:
                    # measure elapsed time
                    printProgressBar(i + 1, self.total + 1,
                                     length=20,
                                     prefix=f'Epoch {epoch} ',
                                     suffix=f', loss: {np.mean(losses):.3f}'
                                     )
            if not opt.SRV:
                printProgressBar(self.total, self.total,
                                 done=f'Epoch [{epoch:>3d}]  d_loss: {np.mean(losses):.4f}'
                                 f', time: {time.time() - start_time:.2f}s'
                                 )
            else:
                print('\nEpoch: ' + str(epoch) + ' | loss: ' + str(np.mean(losses)) + ' | time: ' + str(
                    time.time() - start_time))
            self.eval(write_flag=(self.write_flag and epoch == self.num_epochs - 1))
            self.save()

    def eval(self, write_flag=False):
        with torch.no_grad():
            sigm = nn.Sigmoid()
            self.n.eval()
            losses = []

            start_time = time.time()
            for i, (x, _) in enumerate(self.eval_data_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))
                # compute output
                x = x.to(DEVICE)
                target = x
                output = self.n(x)
                output = sigm(output)
                loss = self.criterion(output, target)
                losses.append(loss.item())
                if write_flag:
                    output = isic.denormalize(output)
                    target = isic.denormalize(target)
                    save_image(output, '/homes/my_d/Autoencoded/out{}.png'.format(i))
                    save_image(target, '/homes/my_d/Autoencoded/tar{}.png'.format(i))

            print('Validation set = Acc: ' + str(1 - np.mean(losses)) + ' | time: ' + str(time.time() - start_time))

    def write_val_images(self, write_flag=False):
        with torch.no_grad():
            self.n.eval()
            losses = []

            start_time = time.time()
            for i, (x, y) in enumerate(self.eval_data_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))
                # compute output
                if i < 2 or i > 4:
                    continue

                x = isic.denormalize(x)
                if i == 2:
                    output = x
                else:
                    output = torch.cat((output,x), 0)
                ry = y.repeat(1, 3, 1, 1)
                output = torch.cat((output, ry),0)

            save_image(output, '/homes/my_d/data/dataset_sample/dataset_whole.png')





    def test(self, write_flag=False):
        with torch.no_grad():

            self.n.eval()
            acc = []
            start_time = time.time()
            for i, (x, target) in enumerate(self.test_data_loader):
                # measure data loading time
                # print("data time: " + str(time.time() - start_time))

                # compute output
                x = x.to(DEVICE)
                target = target.to(DEVICE)
                gt = target.data.squeeze_().cpu().numpy()
                output = self.n(x)
                output = nn.functional.interpolate(output, size=target.shape, mode='bilinear')
                prediction = output.data.squeeze_(1).squeeze_().cpu().numpy()
                acc.append(jaccard(prediction, gt))

            print('Test set = Acc: ' + str(np.mean(acc)) + ' | time: ' + str(time.time() - start_time))
        if write_flag:
            ffname = opt.outd + 'UNet_accuracies.txt'
            with open(ffname, 'a') as f:
                f.write(str(np.mean(acc)) + '\n')

    def save(self):
        try:
            torch.save(self.n.state_dict(), os.path.join(self.outm, self.net_name + self.ckpt_name + '_net.pth'))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.outm, self.net_name + self.ckpt_name + '_opt.pth'))
            print("model weights successfully saved")
        except Exception:
            print("Error during Saving")

    def load(self, ckpt=None):
        if ckpt is not None:
            self.n.load_state_dict(torch.load(os.path.join(opt.outd, opt.outm, self.net_name + ckpt + '_net.pth')))
            # self.optimizer.load_state_dict(torch.load(os.path.join(opt.outd, opt.outm, self.net_name + ckpt + '_opt.pth')))
        else:
            self.n.load_state_dict(
                torch.load(os.path.join(opt.outd, opt.outm, self.net_name + self.ckpt_name + '_net.pth')))
            self.optimizer.load_state_dict(
                torch.load(os.path.join(opt.outd, opt.outm, self.net_name + self.ckpt_name + '_opt.pth')))

    def load_from_path(self, pretrain_path='/homes/my_d/deeplab/deeplab-voc-resnet.pth'):
        # pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrain_dict = torch.load(pretrain_path)
        model_dict = {}
        state_dict = self.n.state_dict()

        new = list(pretrain_dict.get('state_dict').items())

        count = 0
        for key, value in state_dict.items():
            layer_name, weights = new[count]
            state_dict[key] = weights
            count += 1

        state_dict.update(model_dict)
        self.n.load_state_dict(state_dict)
        print("Having loaded voc-pretrained successfully!")


    def find_stats(self):
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in self.data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\ntraining")
        print("mean: " + str(mean) + " | std: " + str(std))

        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in self.eval_data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\nvalidation")
        print("mean: " + str(mean) + " | std: " + str(std))

        mean = 0.
        std = 0.
        nb_samples = 0.
        for data, _ in self.test_data_loader:
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print("\ntest")
        print("mean: " + str(mean) + " | std: " + str(std))


if __name__ == '__main__':

    n = SegmentationNet(num_epochs=opt.epochs, l_r=opt.learning_rate, size=opt.imageSize, thresh=opt.thresh, n_workers=opt.workers,
                        batch_size=opt.batch_size, net_name=opt.network, ckpt_name=opt.ckpt_name)
    if opt.load_ckpt_name is not None:
        n.load(opt.load_ckpt_name)
    if opt.network == "DeepLab" and opt.ckpt_name[:4] == 'voc':
        n.load_from_path()
    # n.train(opt.epochs - opt.loadEpoch)
    # n.eval(True)
    # if opt.network == "DeepLab" and opt.dataset == 'isic':
    #     n.learning_rate = n.learning_rate / 7
    #     n.n.freeze_bn()
    #     n.n.change_o_stride(8)
    #     n.n.to(DEVICE)
    #     n.train()
    # #
    # n.test()
    # n.save()
    n.write_val_images()
# -*- coding: utf-8 -*-
# @Time    : 2018/9/26 15:48
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train_voc.py
# @Software: PyCharm

import os
import pprint
import logging
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from math import ceil
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from PIL import Image, ImageOps
from torchvision import transforms
import cv2 as cv
import sys
sys.path.append(os.path.abspath('..'))
from graphs.models.sync_batchnorm.replicate import patch_replication_callback
from eval import Eval
from graphs.models.decoder import DeepLab
from datasets.isic_dataset_all import ISICDataLoader
from graphs.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import collections
import torchvision.utils as ut

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class tanimoto_loss(torch.nn.Module):
    def __init__(self):
        super(tanimoto_loss, self).__init__()
        self.eps = 1e-07

    def forward(self, output, grounds):
        intersection = torch.sum(output*grounds)
        union_loss = torch.sum(output**2) + torch.sum(grounds**2) - intersection

        loss = (intersection + self.eps) / (union_loss + self.eps)
        loss = 1 - loss

        return loss

class Trainer():
    def __init__(self, args, cuda=None):
        self.args = args
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.current_MIoU = 0
        self.best_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0

        self.batch_idx = 0

        # set TensorboardX
        self.writer = SummaryWriter()

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        if self.args.loss == 'tanimoto':
            self.loss = tanimoto_loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

        self.loss.to(self.device)

        # model
        self.model = DeepLab(output_stride=self.args.output_stride,
                             class_num=self.args.num_classes,
                             num_input_channel=self.args.input_channels,
                             pretrained=self.args.imagenet_pretrained and self.args.pretrained_ckpt_file is None,
                             bn_eps=self.args.bn_eps,
                             bn_momentum=self.args.bn_momentum,
                             freeze_bn=self.args.freeze_bn)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            patch_replication_callback(self.model)
            self.m = self.model.module
        else:
            self.m = self.model
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(
            params=[
                {
                    "params": self.get_params(self.m, key="1x"),
                    "lr": self.args.lr,
                },
                {
                    "params": self.get_params(self.m, key="10x"),
                    "lr": 10 * self.args.lr,
                },
            ],
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        self.dataloader = ISICDataLoader(self.args)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.train_iterations)

        if self.args.input_channels == 3:
            self.train_func = self.train_3ch
            if args.using_bb != 'none':
                if self.args.store_result:
                    self.validate_func = self.validate_crop_store_result
                else:
                    self.validate_func = self.validate_crop
            else:
                self.validate_func = self.validate_3ch
        else:
            self.train_func = self.train_4ch
            self.validate_func = self.validate_4ch

        if self.args.store_result:
            self.validate_one_epoch = self.validate_one_epoch_store_result

    def main(self):
        logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            logger.info("{:16} {}".format(key, val))

        if self.cuda:
            current_device = torch.cuda.current_device()
            logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            logger.info("This model will run on CPU")

        if self.args.pretrained_ckpt_file is not None:
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        if self.args.validate:
            self.validate()
        else:
            self.train()

        self.writer.close()

    def train(self):
        for epoch in tqdm(range(self.current_epoch, self.epoch_num),
                          desc="Total {} epochs".format(self.epoch_num)):
            self.current_epoch = epoch
            tqdm_epoch = tqdm(self.dataloader.train_loader, total=self.dataloader.train_iterations,
                              desc="Train Epoch-{}-".format(self.current_epoch + 1))
            logger.info("Training one epoch...")
            self.Eval.reset()

            self.train_loss = []
            self.model.train()
            if self.args.freeze_bn:
                for m in self.model.modules():
                    if isinstance(m, SynchronizedBatchNorm2d):
                        m.eval()

            # Initialize your average meters
            self.train_func(tqdm_epoch)

            MIoU_single_img, MIoU_thresh = self.Eval.Mean_Intersection_over_Union()

            logger.info('Epoch:{}, train MIoU1:{}'.format(self.current_epoch, MIoU_thresh))
            tr_loss = sum(self.train_loss) / len(self.train_loss)
            self.writer.add_scalar('train_loss', tr_loss, self.current_epoch)
            tqdm.write("The average loss of train epoch-{}-:{}".format(self.current_epoch, tr_loss))
            tqdm_epoch.close()

            if self.current_epoch % 10 == 0:
                state = {
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_MIou': self.current_MIoU
                }
                # logger.info("=>saving the final checkpoint...")
                torch.save(state, train_id + '_epoca_' + str(self.current_epoch))

            # validate
            if self.args.validation:
                MIoU, MIoU_thresh = self.validate()
                self.writer.add_scalar('MIoU', MIoU_thresh, self.current_epoch)

                self.current_MIoU = MIoU_thresh
                is_best = MIoU_thresh > self.best_MIou
                if is_best:
                    self.best_MIou = MIoU_thresh
                self.save_checkpoint(is_best, train_id + 'best.pth')

        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.current_MIoU
        }
        logger.info("=>saving the final checkpoint...")
        torch.save(state, train_id + 'final.pth')

    def train_3ch(self, tqdm_epoch):
        for x, y in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            if self.current_iter >= self.args.iter_max:
                logger.info("iteration arrive {}!".format(self.args.iter_max))
                break
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)
            self.writer.add_scalar('learning_rate_10x', self.optimizer.param_groups[1]["lr"], self.current_iter)
            self.train_one_epoch(x, y)

    def train_4ch(self, tqdm_epoch):
        for x, y, target in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            if self.current_iter >= self.args.iter_max:
                logger.info("iteration arrive {}!".format(self.args.iter_max))
                break
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)
            self.writer.add_scalar('learning_rate_10x', self.optimizer.param_groups[1]["lr"], self.current_iter)

            target = target.float()
            x = torch.cat((x, target), dim=1)
            self.train_one_epoch(x, y)

    def train_one_epoch(self, x, y):
        if self.cuda:
            x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

        y[y > 0] = 1.
        self.optimizer.zero_grad()

        # model
        pred = self.model(x)

        y = torch.squeeze(y, 1)
        if self.args.num_classes == 1:
            y = y.to(device=self.device, dtype=torch.float)
            pred = pred.squeeze()
        # loss
        cur_loss = self.loss(pred, y)

        # optimizer
        cur_loss.backward()
        self.optimizer.step()

        self.train_loss.append(cur_loss.item())

        if self.batch_idx % 50 == 0:
            logger.info("The train loss of epoch{}-batch-{}:{}".format(self.current_epoch,
                                                                       self.batch_idx, cur_loss.item()))
        self.batch_idx += 1

        self.current_iter += 1

        # print(cur_loss)
        if np.isnan(float(cur_loss.item())):
            raise ValueError('Loss is nan during training...')

    def validate(self):
        logger.info('validating one epoch...')
        self.Eval.reset()
        self.iter = 0

        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.valid_loader, total=self.dataloader.valid_iterations,
                              desc="Val Epoch-{}-".format(self.current_epoch + 1))
            self.val_loss = []
            self.model.eval()
            self.validate_func(tqdm_batch)

            MIoU, MIoU_thresh = self.Eval.Mean_Intersection_over_Union()

            logger.info('validation MIoU1:{}'.format(MIoU))
            v_loss = sum(self.val_loss) / len(self.val_loss)
            print('Miou: ' + str(MIoU) + ' MIoU_thresh: ' + str(MIoU_thresh))

            self.writer.add_scalar('val_loss', v_loss, self.current_epoch)

            tqdm_batch.close()

        return MIoU, MIoU_thresh

    def validate_3ch(self,tqdm_batch):
        for x, y, w, h, name in tqdm_batch:
            self.validate_one_epoch(x, y, w, h, name)

    def validate_4ch(self, tqdm_batch):
        for x, y, target, w, h, name in tqdm_batch:
            target = target.float()
            x = torch.cat((x, target), dim=1)
            self.validate_one_epoch(x, y, w, h, name)

    def validate_crop(self, tqdm_batch):
        for i, (x, y, left, top, right, bottom, w, h, name) in enumerate(tqdm_batch):
            self.validate_one_epoch(x, y, w, h, name, left, top, right, bottom)

    def validate_crop_store_result(self, tqdm_batch):
        for i, (x, y, left, top, right, bottom, w, h, name) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

            # model
            pred = self.model(x)
            if self.args.loss == 'tanimoto':
                pred = (pred - pred.min()) / (pred.max() - pred.min())
            else:
                pred = nn.Sigmoid()(pred)

            pred = pred.squeeze().data.cpu().numpy()
            for i, single_argpred in enumerate(pred):
                pil = Image.fromarray(single_argpred)
                pil = pil.resize((right[i] - left[i], bottom[i] - top[i]))
                img = np.array(pil)
                img_border = cv.copyMakeBorder(img, top[i].numpy(), h[i].numpy() - bottom[i].numpy(),
                                               left[i].numpy(),
                                               w[i].numpy() - right[i].numpy(), cv.BORDER_CONSTANT, value=[0, 0, 0])

                if self.args.store_result:
                    img_border *= 255
                    pil = Image.fromarray(img_border.astype('uint8'))
                    pil.save(args.result_filepath + 'ISIC_{}.png'.format(name[i]))

                    self.iter += 1

    def validate_one_epoch_store_result(self, x, y, w, h, name):
        if self.cuda:
            x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

        # model
        pred = self.model(x)
        if self.args.loss == 'tanimoto':
            pred = (pred - pred.min()) / (pred.max() - pred.min())
        else:
            pred = nn.Sigmoid()(pred)

        pred = pred.squeeze().data.cpu().numpy()
        for i, single_argpred in enumerate(pred):
            pil = Image.fromarray(single_argpred)
            pil = pil.resize((w[i], h[i]))
            img_border = np.array(pil)
            if self.args.store_result:
                img_border *= 255
                pil = Image.fromarray(img_border.astype('uint8'))
                pil.save(args.result_filepath + 'ISIC_{}.png'.format(name[i]))

                self.iter += 1

    # def validate_crop(self, tqdm_batch):
    #     for i, (x, y, left, top, right, bottom, w, h, name) in enumerate(tqdm_batch):
    #         if self.cuda:
    #             x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
    #
    #         pred = self.model(x)
    #         y = torch.squeeze(y, 1)
    #         if self.args.num_classes == 1:
    #             y = y.to(device=self.device, dtype=torch.float)
    #             pred = pred.squeeze()
    #
    #         cur_loss = self.loss(pred, y)
    #         if np.isnan(float(cur_loss.item())):
    #             raise ValueError('Loss is nan during validating...')
    #         self.val_loss.append(cur_loss.item())
    #
    #         pred = pred.data.cpu().numpy()
    #
    #         pred[pred >= 0.5] = 1
    #         pred[pred < 0.5] = 0
    #         print('\n')
    #         for i, single_pred in enumerate(pred):
    #             gt = Image.open(self.args.data_root_path + "ground_truth/ISIC_" + name[i] + "_segmentation.png")
    #             pil = Image.fromarray(single_pred.astype('uint8'))
    #             pil = pil.resize((right[i] - left[i], bottom[i] - top[i]))
    #             img = np.array(pil)
    #             ground_border = np.array(gt)
    #             ground_border[ground_border == 255] = 1
    #             img_border = cv.copyMakeBorder(img, top[i].numpy(), h[i].numpy() - bottom[i].numpy(),
    #                                            left[i].numpy(),
    #                                            w[i].numpy() - right[i].numpy(), cv.BORDER_CONSTANT, value=[0, 0, 0])
    #
    #             iou = self.Eval.iou_numpy(img_border, ground_border)
    #             print(name[i] + ' iou: ' + str(iou))
    #
    #             if self.args.store_result:
    #                 img_border[img_border == 1] = 255
    #                 pil = Image.fromarray(img_border)
    #                 pil.save(args.result_filepath + 'ISIC_{}.png'.format(name[i]))
    #                 # gt.save(args.result_filepath + 'ISIC_ground_{}.png'.format(name[i]))
    #
    #                 self.iter += 1

    def validate_one_epoch(self, x, y, w, h, name, *ltrb):
        if self.cuda:
            x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

        # model
        pred = self.model(x)
        y = torch.squeeze(y, 1)
        if self.args.num_classes == 1:
            y = y.to(device=self.device, dtype=torch.float)
            pred = pred.squeeze()

        cur_loss = self.loss(pred, y)
        if np.isnan(float(cur_loss.item())):
            raise ValueError('Loss is nan during validating...')
        self.val_loss.append(cur_loss.item())

        pred = pred.data.cpu().numpy()

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        print('\n')
        for i, single_pred in enumerate(pred):
            gt = Image.open(self.args.data_root_path + "ground_truth/ISIC_" + name[i] + "_segmentation.png")
            pil = Image.fromarray(single_pred.astype('uint8'))

            if self.args.using_bb and self.args.input_channels==3:
                pil = pil.resize((ltrb[2][i] - ltrb[0][i], ltrb[3][i] - ltrb[1][i]))
                img = np.array(pil)
                img_border = cv.copyMakeBorder(img, ltrb[1][i].numpy(), h[i].numpy() - ltrb[3][i].numpy(),
                                               ltrb[0][i].numpy(),
                                               w[i].numpy() - ltrb[2][i].numpy(), cv.BORDER_CONSTANT, value=[0, 0, 0])
            else:
                pil = pil.resize((w[i], h[i]))
                img_border = np.array(pil)

            ground_border = np.array(gt)
            ground_border[ground_border == 255] = 1
            iou = self.Eval.IoU_one_class(img_border, ground_border)

            print(name[i] + ' iou: ' + str(iou))

            if self.args.store_result:
                img_border[img_border == 1] = 255
                pil = Image.fromarray(img_border)
                pil.save(args.result_filepath + 'ISIC_{}.png'.format(name[i]))
                # gt.save(args.result_filepath + 'ISIC_ground_{}.png'.format(name[i]))

                self.iter += 1

    def save_checkpoint(self, is_best, filename=None):
        filename = os.path.join(self.args.checkpoint_dir, filename)
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_MIou': self.best_MIou
        }
        if is_best:
            logger.info("=>saving a new best checkpoint...")
            torch.save(state, filename)
        else:
            logger.info("=> The MIoU of val does't improve.")

    def load_checkpoint(self, filename):
        try:
            logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            if 'module.Resnet101.bn1.weight' in checkpoint['state_dict']:
                checkpoint2 = collections.OrderedDict([(k[7:], v) for k, v in checkpoint['state_dict'].items()])
                self.model.load_state_dict(checkpoint2)
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            if not self.args.freeze_bn:
                self.current_epoch = checkpoint['epoch']
                self.current_iter = checkpoint['iteration']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_MIou = checkpoint['best_MIou']
            print("Checkpoint loaded successfully from '{}', MIoU:{})\n"
             .format(self.args.checkpoint_dir, checkpoint['best_MIou']))
            logger.info("Checkpoint loaded successfully from '{}', MIoU:{})\n"
                  .format(self.args.checkpoint_dir, checkpoint['best_MIou']))
        except OSError as e:
            logger.info("No checkpoint exists from '{}'. Skipping...".format(self.args.checkpoint_dir))
            logger.info("**First time to train**")

    def get_params(self, model, key):
        # For Dilated CNN
        if key == "1x":
            for m in model.named_modules():
                if "Resnet101" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
        #
        if key == "10x":
            for m in model.named_modules():
                if "encoder" in m[0] or "decoder" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p

    def poly_lr_scheduler(self, optimizer, init_lr, iter, max_iter, power):
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        optimizer.param_groups[0]["lr"] = new_lr
        optimizer.param_groups[1]["lr"] = 10 * new_lr

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    arg_parser = argparse.ArgumentParser()

    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default="/path/to/data/ISIC_dataset/Task_1/",
                            help="the root path of the dataset")
    arg_parser.add_argument('--checkpoint_dir', default="/path/of/ckpt/",
                            help="the path of checkpoint files")
    arg_parser.add_argument('--result_filepath', default="/path/to/save/images/",
                            help="the filepath where masks are stored")
    arg_parser.add_argument('--bb_path', default="/path/to/maskrcnn/predictions.json")
    arg_parser.add_argument('--outname', type=str, default="_")
    arg_parser.add_argument('--input_channels', type=int, default=3, choices=[3, 4],
                            help="choose from 3, 4 with bounding box")

    # Model related arguments
    arg_parser.add_argument('--backbone', default='resnet101',
                            help="backbone of encoder")
    arg_parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16],
                            help="choose from 8 or 16")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply iamgenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--save_ckpt_file', type=str2bool, default=True,
                            help="whether to save trained checkpoint file ")
    arg_parser.add_argument('--store_result', type=str2bool, default=True,
                            help="whether to store masks after val or test")
    arg_parser.add_argument('--validation', type=str2bool, default=True,
                            help="whether to val after each train epoch")
    arg_parser.add_argument('--using_bb', type=str2bool, default=True,
                            help="whether to use bounding box ")
    arg_parser.add_argument('--validate', type=str2bool, default=False,
                            help="whether validate or train")

    # train related arguments
    arg_parser.add_argument('--gpu', type=str, default="1,3",
                            help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                            help='input batch size')
    arg_parser.add_argument('--batch_size', default=12, type=int,
                            help='input batch size')

    # dataset related arguments
    arg_parser.add_argument('--base_size', default=513, type=int,
                            help='crop size of image')
    arg_parser.add_argument('--crop_size', default=513, type=int,
                            help='base size of image')
    arg_parser.add_argument('--num_classes', default=1, type=int,
                            help='num classes of masks')
    arg_parser.add_argument('--data_loader_workers', default=8, type=int,
                            help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    arg_parser.add_argument('--train_split', type=str, default='training_2017',
                            help="choose the split to training")
    arg_parser.add_argument('--val_split', type=str, default='validation_2017',
                            help="choose the split to validate/test")
    arg_parser.add_argument('--loss', type=str, default='ce',
                            choices=['ce', 'tanimoto'])
    # optimization related arguments

    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--bn_eps', type=float, default=1e-05)
    arg_parser.add_argument('--weight_decay', type=float, default=4e-5)

    arg_parser.add_argument('--lr', type=float, default=0.007,
                            help="initial learning rate")
    arg_parser.add_argument('--iter_max', type=int, default=30000,
                            help="the maximum of iteration")
    arg_parser.add_argument('--poly_power', type=float, default=0.9,
                            help="poly_power")
    args = arg_parser.parse_args()

    train_id = str(args.backbone) + '_' + str(args.output_stride)
    train_id += '_iamgenet_pre-' + str(args.imagenet_pretrained)
    train_id += '_batch_size-' + str(args.batch_size)
    train_id += '_train_split-' + str(args.train_split)
    train_id += '_val_split-' + str(args.val_split)
    train_id += '_lr-' + str(args.lr)
    train_id += '_iter_max-' + str(args.iter_max)
    train_id += str(args.outname)

    # logger configure
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(train_id+'.txt')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    agent = Trainer(args=args, cuda=True)
    agent.main()

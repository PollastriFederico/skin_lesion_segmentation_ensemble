from __future__ import print_function
from PIL import Image, ImageOps, ImageFilter
import os
import os.path
import time
import random
from random import randint
import torch
import numpy as np
from torchvision import transforms, utils
import transforms as transforms_mod
import transforms_4ch
import torch.utils.data as data
import cv2 as cv
import json
import math
import torchvision.utils as ut

class ISIC_Dataset(data.Dataset):
    data_root = '/homes/my_d/data/ISIC_dataset/Task_1/'
    splitsdic = {
        'training_2017': data_root + "splits/2017_training.csv",
        'validation_2017': data_root + "splits/2017_validation.csv",
        'test_2017': data_root + "splits/2017_test.csv",
        'training_2018': data_root + "splits/2018_training.csv",
        'validation_2018': data_root + "splits/2018_validation.csv",
        'test_2018': data_root + "splits/2018_test.csv",
        'from_API': data_root + "splits/from_API.csv",
        'dermoscopic': data_root + "splits/dermoscopic.csv",
        'clinic': data_root + "splits/clinic.csv",
        'dermoscopic_wmasks': data_root + "splits/dermoscopic_with_mask.csv",
        'dermot_2017': data_root + "splits/dermoscopic_train_2017.csv",
        'mtap': data_root + "splits/dermo_MTAP.csv",

    }

    def __init__(self, root, batch_size=8, split_list=None, split_name='training_2018', size=(513, 513),
                 bb_path='/path/to/json/prediction.json', bb=True, is_training=True,
                 segmentation_transform=None, transform=None, target_transform=None):
        start_time = time.time()
        self.root = root
        self.segmentation_transform = segmentation_transform
        self.transform = transform
        self.target_transform = target_transform
        self.split_list = split_list
        self.size = size
        self.crop_size = size[0]
        self.base_size = size[0]
        self.bb_path = bb_path
        self.img_size = []
        self.batch_size = batch_size
        self.split_name = split_name
        self.bb = bb
        self.is_training = is_training

        if self.bb:
            with open(self.bb_path) as f_e:
                self.list_pred_eval = json.load(f_e)
            f_e.close()

        if split_list is None:
            print('loading ' + split_name)
            self.split_list = self.read_csv(split_name)

        print("Time: " + str(time.time() - start_time))

    def cntrl(self, left, top, right, bottom, image_size):
        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if right > image_size[0]:
            right = image_size[0]
        if bottom > image_size[1]:
            bottom = image_size[1]
        return left, top, right, bottom

    def img_crop(self, image, ground, index):
        if self.is_training:
            target = np.array(ground).astype('int32')
            foreground_pixels = np.array(np.where(target == 255))
            top = min(foreground_pixels[0, :])
            bottom = max(foreground_pixels[0, :])
            left = min(foreground_pixels[1, :])
            right = max(foreground_pixels[1, :])
            ground = ground.crop((left, top, right, bottom))
            image = image.crop((left, top, right, bottom))
            return image, ground, left, top, right, bottom
        else:
            if self.list_pred_eval[index] == [0, 0, 0, 0]:
                left_pred = 0
                top_pred = 0
                right_pred = image.size[0]
                bottom_pred = image.size[1]
            else:
                left_pred = self.list_pred_eval[index][0]
                top_pred = self.list_pred_eval[index][1]
                right_pred = self.list_pred_eval[index][2]
                bottom_pred = self.list_pred_eval[index][3]
                left_pred, top_pred, right_pred, bottom_pred = self.cntrl(left_pred, top_pred, right_pred, bottom_pred, image.size)
            ground = ground.crop((left_pred, top_pred, right_pred, bottom_pred))
            image = image.crop((left_pred, top_pred, right_pred, bottom_pred))
            return image, ground, left_pred, top_pred, right_pred, bottom_pred

    def img_4ch(self, w, h, ground, index):
        if self.is_training:
            target = np.array(ground).astype('int32')
            foreground_pixels = np.array(np.where(target == 255))
            top = min(foreground_pixels[0, :])
            bottom = max(foreground_pixels[0, :])
            left = min(foreground_pixels[1, :])
            right = max(foreground_pixels[1, :])
            cont = np.array([[left, top], [left, bottom], [right, bottom], [right, top]])
            target = cv.fillPoly(target, pts=[cont], color=1)
            target = Image.fromarray(target)
            return target
        else:
            list_pred = self.list_pred_eval
            target = Image.new('L', (w, h), 0)
            target = np.array(target).astype('int32')
            if list_pred[index] == [0, 0, 0, 0]:
                left = 0
                top = 0
                right = w
                bottom = h
            else:
                left = list_pred[index][0]
                top = list_pred[index][1]
                right = list_pred[index][2]
                bottom = list_pred[index][3]
                left, top, right, bottom = self.cntrl(left, top, right, bottom, (w, h))

        cont = np.array([[left, top], [left, bottom], [right, bottom], [right, top]])
        target = cv.fillPoly(target, pts=[cont], color=1)
        target = Image.fromarray(target)
        return target

    def get_img_info(self, index):
        image = Image.open(self.imgs[index])
        width, height = image.size
        return width, height

    def get_names(self, i_list):
        imgs = []
        grnds = []
        for i in i_list:
            imgs.append(self.root + "images/ISIC_" + str(i) + ".jpg")
            grnds.append(self.root + "ground_truth/ISIC_" + str(i) + "_segmentation.png")
        return imgs, grnds

    @classmethod
    def get_images(cls, i_list, size):
        imgs = []
        grnds = []
        for i in i_list:
            imgs.append(Image.open(cls.data_root + "images/ISIC_" + str(i) + ".jpg").resize(size, Image.BICUBIC))
            grnds.append(Image.open(cls.data_root + "ground_truth/ISIC_" + str(i) + "_segmentation.png").resize(size,
                                                                                                                Image.BICUBIC))
        return imgs, grnds

    @classmethod
    def read_csv(cls, csv_filename):
        import csv
        split_list = []
        with open(cls.splitsdic.get(csv_filename)) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                split_list.append(row[0])

        return split_list

    def __len__(self):
        return len(self.split_list)

class ISIC_Dataset_3ch(ISIC_Dataset):
    def __init__(self, root, split_name, bb_path, bb, is_training, transform, target_transform, segmentation_transform):
        super(ISIC_Dataset_3ch, self).__init__(root=root, split_name=split_name, bb_path=bb_path,
                                bb=bb, is_training=is_training,
                                transform=transform, target_transform=target_transform, segmentation_transform = segmentation_transform)
        self.imgs, self.grnds = self.get_names(self.split_list)

    def transforms(self, image, ground):
        if self.segmentation_transform is not None:
            image, ground = self.segmentation_transform(image, ground)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            ground = self.target_transform(ground)
        return image, ground

    def __getitem__(self, index):
        image_orig = Image.open(self.imgs[index])
        ground_orig = Image.open(self.grnds[index])
        w = image_orig.size[0]
        h = image_orig.size[1]

        if self.bb:
            image, ground, left, top, right, bottom = self.img_crop(image_orig, ground_orig, index)
            image, ground = self.transforms(image, ground)

            if self.is_training:
                return image, ground
            else:
                return image, ground, left, top, right, bottom, w, h, self.split_list[index]

        image, ground = self.transforms(image_orig, ground_orig)
        if self.is_training:
            return image, ground
        else:
            return image, ground, w, h, self.split_list[index]

class ISIC_Dataset_3ch_noground(ISIC_Dataset_3ch):
    def __init__(self, root, split_name, bb_path, bb, is_training, transform, target_transform, segmentation_transform):
        super(ISIC_Dataset_3ch_noground, self).__init__(root=root, split_name=split_name, bb_path=bb_path,
                                bb=bb, is_training=is_training, transform=transform,
                                target_transform=target_transform, segmentation_transform = segmentation_transform)
        self.imgs, self.grnds = self.get_names(self.split_list)

    def get_names(self, i_list):
        imgs = []
        grnds = []
        for j, i in enumerate(i_list):
            imgs.append(self.root + "images/ISIC_" + str(i) + ".jpg")
            grnds.append('/homes/img_test_maskrcnn_2018/ISIC_' + str(i) + ".png")
        return imgs, grnds

class ISIC_Dataset_4ch(ISIC_Dataset):
    def __init__(self, root, split_name, bb_path, bb, is_training, transform, target_transform, segmentation_transform):
        super(ISIC_Dataset_4ch, self).__init__(root=root, split_name=split_name, bb_path=bb_path,
                                bb=bb, is_training=is_training, transform=transform, target_transform=target_transform,
                                segmentation_transform = segmentation_transform)
        self.imgs, self.grnds = self.get_names(self.split_list)

    def transforms(self, image, ground, target):
        if self.segmentation_transform is not None:
            image, ground, target = self.segmentation_transform(image, ground, target)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            ground = self.target_transform(ground)
            target = self.target_transform(target)
        return image, ground, target

    def __getitem__(self, index):
        image_orig = Image.open(self.imgs[index])
        ground_orig = Image.open(self.grnds[index])
        w = image_orig.size[0]
        h = image_orig.size[1]

        target = self.img_4ch(w, h, ground_orig, index)
        image, ground, target = self.transforms(image_orig, ground_orig, target)
        if self.is_training:
            return image, ground, target
        else:
            return image, ground, target, w, h, self.split_list[index]

class ISIC_Dataset_4ch_noground(ISIC_Dataset_4ch):
    def __init__(self, root, split_name, bb_path, bb, is_training, transform, target_transform, segmentation_transform):
        super(ISIC_Dataset_4ch_noground, self).__init__(root=root, split_name=split_name, bb_path=bb_path,
                                bb=bb, is_training=is_training, transform=transform, target_transform=target_transform,
                                segmentation_transform = segmentation_transform)
        self.imgs, self.grnds = self.get_names(self.split_list)

    def get_names(self, i_list):
        imgs = []
        grnds = []
        for j, i in enumerate(i_list):
            imgs.append(self.root + "images/ISIC_" + str(i) + ".jpg")
            grnds.append('/homes/img_test_maskrcnn_2018/ISIC_' + str(i) + ".png")
        return imgs, grnds

class ISICDataLoader:
    def __init__(self, args):
        self.args = args
        imagesize = (513, 513)
        if self.args.input_channels == 3:
            tr = transforms_mod
            isic = ISIC_Dataset_3ch
            # isic = ISIC_Dataset_3ch_noground
        else:
            tr = transforms_4ch
            isic = ISIC_Dataset_4ch
            # isic = ISIC_Dataset_4ch_noground

        segmentation_trasform = tr.Compose([
                                    tr.Resize(imagesize),
                                    tr.RandomHorizontalFlip(),
                                    tr.RandomVerticalFlip(),
                                    tr.RandomApply(
                                        [tr.Resize(imagesize),
                                         tr.ColorJitter(0.3, 0.3, 0.2, 0.01),
                                         tr.RandomAffine(degrees=0, shear=5),
                                         tr.RandomRotation(180),
                                         tr.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                                         tr.RandomAffine(degrees=0, scale=(0.95, 1.25))], p=0.4),
                                    tr.ToTensor(),
                                    tr.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351)),
                                ])

        transform = transforms.Compose([
                                   transforms.Resize(imagesize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351))
                               ])

        target_transform = transforms.Compose([
                                   transforms.Resize(imagesize),
                                   transforms.ToTensor()
                               ])

        self.train_set = isic(root=self.args.data_root_path, split_name=args.train_split, bb_path=args.bb_path,
                                bb = self.args.using_bb, is_training=True,
                                transform=None, target_transform=None, segmentation_transform=segmentation_trasform)
        self.val_set = isic(root=self.args.data_root_path, split_name=args.val_split, bb_path = args.bb_path,
                               bb = self.args.using_bb, is_training=False,
                               transform=transform, target_transform=target_transform, segmentation_transform=None)

        self.train_loader = data.DataLoader(self.train_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=True,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=True,
                                            drop_last=True)
        self.valid_loader = data.DataLoader(self.val_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=True,
                                            drop_last=False)

        self.train_iterations = (len(self.train_set) + self.args.batch_size) // self.args.batch_size
        self.valid_iterations = (len(self.val_set) + self.args.batch_size) // self.args.batch_size

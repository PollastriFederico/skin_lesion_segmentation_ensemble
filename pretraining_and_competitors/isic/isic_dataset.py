from __future__ import print_function
from PIL import Image
import os
import os.path
import time

import torch.utils.data as data

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


class ISIC(data.Dataset):
    """ ISIC Dataset. """

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

    def __init__(self, root, split_list=None, split_name='training_2018', load=False, size=(512, 512),
                 segmentation_transform=None, transform=None, target_transform=None):
        start_time = time.time()
        self.root = os.path.expanduser(root)
        self.segmentation_transform = segmentation_transform
        self.transform = transform
        self.target_transform = target_transform
        self.split_list = split_list
        self.load = load
        self.size = size
        if split_list is None:
            print('loading ' + split_name)
            self.split_list = self.read_csv(split_name)

            if load:
                print("LOADING " + str(len(self.split_list)) + " images in MEMORY")
                self.imgs, self.grnds = self.get_images(self.split_list, self.size)
            else:
                self.imgs, self.grnds = self.get_names(self.split_list)

        else:
            self.imgs = self.split_list
            self.grnds = self.split_list

        print("Time: " + str(time.time() - start_time))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, ground)
        """

        if not self.load:
            image = Image.open(self.imgs[index])
            try:
                ground = Image.open(self.grnds[index])
            except:
                # print("you are using images you don't have the segmentation masks for")
                ground = image.convert('L')
                # SOME SEGMENTATION MASKS ARE YET TO BE RELEASED
        else:
            image = self.imgs[index]
            ground = self.grnds[index]

        if self.segmentation_transform is not None:
            image, ground = self.segmentation_transform(image, ground)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            ground = self.target_transform(ground)

        return image, ground

    def __len__(self):
        return len(self.split_list)

    @classmethod
    def get_names(cls, i_list):
        imgs = []
        grnds = []
        for i in i_list:
            imgs.append(cls.data_root + "images/ISIC_" + str(i) + ".jpg")
            grnds.append(cls.data_root + "ground_truth/ISIC_" + str(i) + "_segmentation.png")

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


def denormalize(img):
    mean = [0.3359, 0.1133, 0.0276]
    std = [0.3324, 0.3247, 0.3351]

    for i in range(img.shape[1]):
        img[:, i, :, :] = img[:, i, :, :] * std[i]
        img[:, i, :, :] = img[:, i, :, :] + mean[i]
    return img

    # def __repr__(self):
    #     fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    #     fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    #     tmp = 'train' if self.train is True else 'test'
    #     fmt_str += '    Split: {}\n'.format(tmp)
    #     fmt_str += '    Root Location: {}\n'.format(self.root)
    #     tmp = '    Transforms (if any): '
    #     fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    #     tmp = '    Target Transforms (if any): '
    #     fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    #     return fmt_str

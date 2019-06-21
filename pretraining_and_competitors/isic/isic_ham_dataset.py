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

    data_root = '/homes/my_d/data/ISIC_dataset/Task_3/'
    splitsdic = {
        'training_2018': data_root + "ISIC2018_Task3_Training_GroundTruth.csv",
    }

    def __init__(self, split_list=None, split_name='training_2018', load=False, size=(512, 512),
                 segmentation_transform=None, transform=None, target_transform=None):
        start_time = time.time()
        self.segmentation_transform = segmentation_transform
        self.transform = transform
        self.target_transform = target_transform
        self.split_list = split_list
        self.load = load
        self.size = size

        print('loading ' + split_name)
        self.split_list, self.lbls = self.read_csv(split_name)

        if load:
            print("LOADING " + str(len(self.split_list)) + " images in MEMORY")
            self.imgs = self.get_images(self.split_list, self.size)
        else:
            self.imgs = self.get_names(self.split_list)


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

        else:
            image = self.imgs[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, self.lbls[index], self.imgs[index]

    def __len__(self):
        return len(self.split_list)

    @classmethod
    def get_names(cls, n_list):
        imgs = []
        for n in n_list:
            imgs.append(cls.data_root + "images/" + n + ".jpg")
        return imgs

    @classmethod
    def get_images(cls, i_list, size):
        imgs = []
        for i in i_list:
            imgs.append(Image.open(cls.data_root + "images/ISIC_" + str(i) + ".jpg").resize(size, Image.BICUBIC))
        return imgs

    @classmethod
    def read_csv(cls, csv_filename):
        import csv
        split_list = []
        labels_list = []
        fname = cls.splitsdic.get(csv_filename)

        with open(fname) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                if row[0] == 'image':
                    continue
                split_list.append(row[0])
                for i in range(7):
                    if row[1 + i] == '1.0':
                        labels_list.append(i)
                        break

        return split_list, labels_list


def denormalize(img):
    # mean = [0.3359, 0.1133, 0.0276]
    # std = [0.3324, 0.3247, 0.3351]
    mean = [0.1224, 0.1224, 0.1224]
    std = [0.0851, 0.0851, 0.0851]

    for i in range(img.shape[0]):
        img[i, :, :] = img[i, :, :] * std[i]
        img[i, :, :] = img[i, :, :] + mean[i]
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

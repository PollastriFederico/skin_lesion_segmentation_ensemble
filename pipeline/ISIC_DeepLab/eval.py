# -*- coding: utf-8 -*-
# @Time    : 2018/10/8 13:30
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : eval.py
# @Software: PyCharm

import numpy as np

np.seterr(divide='ignore', invalid='ignore')

class Eval():
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.iou_list = []
        self.iou_list_thresh = []
        self.single_IoU = 0
        self.eps = 1e-06

    def IoU_one_class(self, outputs, labels):
        intersection = (outputs & labels).sum()
        union = (outputs | labels).sum()

        iou = (intersection + self.eps) / (union + self.eps)

        if iou >= 0.65:
            self.iou_list_thresh.append(iou)
        else:
            self.iou_list_thresh.append(0)
        self.iou_list.append(iou)

        return iou

    def Mean_Intersection_over_Union(self):
        return np.nanmean(self.iou_list), np.nanmean(self.iou_list_thresh)

    # generate confusion matrix
    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        MIoU = np.nanmean(MIoU[1:])
        
        if MIoU >= 0.65:
            self.iou_list_thresh.append(MIoU)
        else:
            self.iou_list_thresh.append(0)
        self.iou_list.append(MIoU)
        self.single_IoU = MIoU
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert the size of two images are same
        assert gt_image.shape == pre_image.shape

        self.confusion_matrix += self.__generate_matrix(gt_image, pre_image)
        return self.single_IoU

    def reset(self):
        self.iou_list = []
        self.iou_list_thresh = []
        self.single_IoU = 0
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
import my_transforms as tr

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5
        transform = tr.Compose([
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            tr.RandomApply(
                [
                 tr.ColorJitter(0.3, 0.3, 0.2, 0.01),
                ], p=0.4),
            tr.ToTensor(),
            tr.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351), cfg.INPUT.TO_BGR255),
        ])
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        transform = tr.Compose([
            T.Resize(min_size, max_size),
            tr.ToTensor(),
            tr.Normalize((0.3359, 0.1133, 0.0276), (0.3324, 0.3247, 0.3351), cfg.INPUT.TO_BGR255)
        ])
    # to_bgr255 = cfg.INPUT.TO_BGR255
    # normalize_transform = T.Normalize(
    #     mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    # )
    #
    # transform = T.Compose(
    #     [
    #         T.Resize(min_size, max_size),
    #         T.RandomHorizontalFlip(flip_prob),
    #         T.ToTensor(),
    #         normalize_transform,
    #     ]
    # )
    return transform

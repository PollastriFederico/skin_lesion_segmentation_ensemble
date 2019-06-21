from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image
import os
import os.path
import time
import torch
import numpy as np
import torch.utils.data as data
from skimage import measure
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

class ISIC_Dataset(data.Dataset):
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
    CLASSES = (
        "__background__ ",
        "foreground",
    )
    def __init__(self, data_dir, batch_size = 8, split_list=None, split_name='training_2018', load=False, is_training=None, size=(513, 513),
                 bb = False, segmentation_transform=None, transform=None, target_transform=None):
        start_time = time.time()
        self.root = os.path.expanduser(data_dir)
        self.segmentation_transform = segmentation_transform
        self.transform = transform
        self.target_transform = target_transform
        self.split_list = split_list
        self.load = load
        self.size = size
        self.is_training = is_training
        self.crop_size = size[0]
        self.base_size = size[0]
        self.bb = bb
        self.img_size = []
        self.grounds = []
        self.batch_size = batch_size
        self.split_name = split_name

        if split_list is None:
            print('loading ' + split_name)
            self.split_list = self.read_csv(split_name)

        if load:
            print("LOADING " + str(len(self.split_list)) + " images in MEMORY")
            self.imgs, self.grnds = self.get_images(self.split_list, self.size)
        else:
            self.imgs, self.grnds = self.get_names(self.split_list)

        cls = ISIC_Dataset.CLASSES
        self.class_to_ind = dict(zip(cls, enumerate(cls)))

        print("Time: " + str(time.time() - start_time))

    def __getitem__(self, index):
        image = Image.open(self.imgs[index]).convert("RGB")
        ground = Image.open(self.grnds[index])

        target = self.get_groundtruth(index)
        label = torch.as_tensor([1])
        target.add_field("labels", label)

        segmentations = []
        contours = measure.find_contours(np.array(ground, dtype=np.uint8), 0.5)
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segm = contour.ravel().tolist()
            segmentations.append(segm)

        masks = SegmentationMask([segmentations], ground.size)
        target.add_field("masks", masks)
        target = target.clip_to_image(remove_empty=True)

        if self.segmentation_transform is not None:
            image, ground = self.segmentation_transform(image, target)

        return image, ground, index

    def __len__(self):
        return len(self.split_list)

    def get_gt_image(self, index):
        return Image.open(self.grnds[index])

    def get_name(self, index):
        return self.split_list[index]

    def get_image(self, index):
        return Image.open(self.imgs[index])

    def get_groundtruth(self, index):
        ground = Image.open(self.grnds[index])
        target = np.array(ground).astype('int32')
        foreground_pixels = np.array(np.where(target == 255))
        top = min(foreground_pixels[0, :])
        bottom = max(foreground_pixels[0, :])
        left = min(foreground_pixels[1, :])
        right = max(foreground_pixels[1, :])
        image_size = ground.size
        bbox = [[left, top, right, bottom]]

        boxes = BoxList(bbox, image_size)
        label = torch.as_tensor([1])
        boxes.add_field("labels", label)

        return boxes

    def get_batch_size(self):
        return self.batch_size

    def get_len_grnds(self):
        return len(self.grnds)

    def get_img_info(self, index):
        image = Image.open(self.imgs[index])
        width, height = image.size
        return {"height": height, "width": width}

    def map_class_id_to_class_name(self, class_id):
        return ISIC_Dataset.CLASSES[class_id]

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
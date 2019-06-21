# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
import cv2 as cv
import json
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
import pycocotools.mask as mask_util
from eval import Eval

def do_isic_evaluation(dataset, predictions, grounds, output_folder, logger, meters):
    masker = Masker(threshold=0.5, padding=1)
    evaluator = Eval(2)  # 2 = num_classes
    store_result = False
    iou = []
    pred_list = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
        mask = masks[0].squeeze()
        ground = grounds[image_id].resize((image_width, image_height))
        if len(prediction) == 0:
            print('img ' + str(image_id) + ' senza bbox')
            pred_list.append(list([0, 0, 0, 0]))
            iou.append(0)
            continue
        iou_value, bbox_index = boxlist_iou(prediction, ground).squeeze().sort(descending=True)
        if len(iou_value.shape) > 0:
            iou.append(iou_value[0])
        else:
            iou.append(iou_value)

        gt_image = np.array(dataset.get_gt_image(image_id))
        if len(bbox_index.shape) > 0:
            pred_list.append(list(prediction.bbox[bbox_index[0]].long().tolist()))
            mask_image = np.array(mask[bbox_index[0]])
        else:
            pred_list.append(list(prediction.bbox[bbox_index].long().tolist()))
            mask_image = np.array(mask)
        gt_image[gt_image == 255] = 1
        single_iou = evaluator.add_batch(gt_image, mask_image)
        print(dataset.get_name(image_id) + ' iou: ' + str(single_iou))

        if store_result:
            # res = overlay_boxes(gt_image.copy(), pred_boxlists[i].bbox[bbox_index[0]])
            # res = Image.fromarray(res)
            # res.save('box_img_' + str(i) + '.png')
            mask_image[mask_image==1] = 255
            segm = Image.fromarray(mask_image)
            segm.save('{}-out_test.png'.format(image_id))

    MIoU, MIoU_thresh = evaluator.Mean_Intersection_over_Union()

    print('MIoU: ' + str(MIoU) + ' MIoU_thresh: ' + str(MIoU_thresh))

    result_str = 'mIOU: ' + str(np.nanmean(iou)) + ' - '
    result_str += 'IOU: ' + str(iou)

    logger.info(result_str)
    if store_result:
        with open(os.path.join(output_folder, "predictions_test.json"), "w") as fid:
            json.dump(pred_list, fid)
        with open(os.path.join(output_folder, "result.txt"), "a") as fid:
            fid.write("\n" + result_str)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(int(boxA[0]), int(boxB[0]))
    yA = max(int(boxA[1]), int(boxB[1]))
    xB = min(int(boxA[2]), int(boxB[2]))
    yB = min(int(boxA[3]), int(boxB[3]))

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def select_top_prediction(predictions):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    # scores = predictions.get_field("scores")
    # keep = torch.nonzero(scores > 0.7).squeeze(1)
    # predictions = predictions[keep]

    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx[[0]]]

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    # labels = predictions.get_field("labels")
    # boxes = predictions.bbox

    # colors = compute_colors_for_labels(labels).tolist()

    # for box in zip(boxes):
    #     box = box.to(torch.int64)
    #     top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    #     cv.rectangle(
    #         image, tuple(top_left), tuple(bottom_right), (255,255,255), 3
    #     )
    left = predictions[0].long().numpy()
    top = predictions[1].long().numpy()
    right = predictions[2].long().numpy()
    bottom = predictions[3].long().numpy()
    cv.rectangle(
            image, (left, top), (right, bottom), (255,255,255), 3
        )
    return image

def overlay_mask(image, prediction):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    # mask = prediction.get_field("mask").numpy()
    # labels = predictions.get_field("labels")

    # colors = compute_colors_for_labels(1).tolist()

    # for mask, color in zip(masks, colors):
    # thresh = prediction[0, :, :, None]
    prediction = prediction.squeeze().numpy().astype('uint8')
    _, contours, hierarchy = cv.findContours(
        prediction, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    image = cv.drawContours(image, contours, -1, (255,255,255), 3)

    return image

def calc_detection_isic_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_isic_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
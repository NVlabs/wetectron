# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from . import cv2_util
from .checkpoint import DetectronCheckpointer
from ..structures.image_list import to_image_list
from ..modeling.roi_heads.mask_head.inference import Masker
from ..modeling.detector import build_detection_model

from wetectron.config import cfg
from wetectron import layers as L
from wetectron.structures.keypoint import PersonKeypoints


COCO_CATEGORIES = ["__background",
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign",
    "parking meter", "bench", "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier", "toothbrush"]

VOC_CATEGORIES = ["__background", 
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair", "cow","diningtable","dog",
    "horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

def compute_colors_for_labels(labels):
    """ Simple function that adds fixed colors depending on the class """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("bool")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox
    colors = compute_colors_for_labels(labels).tolist()
    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 2
        )
    return image

def overlay_mask(image, predictions, alpha = 0.5):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    temp = image.copy()
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")
    colors = compute_colors_for_labels(labels).tolist()
    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2_util.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, color, -1)
    composite = alpha*image + (1-alpha)*temp
    return composite

def create_mask_montage(image, predictions, masks_per_dim):
    """
    Create a montage showing the probability heatmaps for each one one of the
    detected objects
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask`.
    """
    masks = predictions.get_field("mask")
    masks = L.interpolate(
        masks.float(), scale_factor=1 / masks_per_dim
    ).byte()
    height, width = masks.shape[-2:]
    max_masks = masks_per_dim ** 2
    masks = masks[:max_masks]
    # handle case where we have less detections than max_masks
    if len(masks) < max_masks:
        masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.bool)
        masks_padded[: len(masks)] = masks
        masks = masks_padded
    masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
    result = torch.zeros(
        (masks_per_dim * height, masks_per_dim * width), dtype=torch.bool
    )
    for y in range(masks_per_dim):
        start_y = y * height
        end_y = (y + 1) * height
        for x in range(masks_per_dim):
            start_x = x * width
            end_x = (x + 1) * width
            result[start_y:end_y, start_x:end_x] = masks[y, x]
    return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

def overlay_keypoints(image, predictions):
    keypoints = predictions.get_field("keypoints")
    kps = keypoints.keypoints
    scores = keypoints.get_field("logits")
    kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
    for region in kps:
        image = vis_keypoints(image, region.transpose((1, 0)))
    return image

def overlay_class_names(image, predictions, CATEGORIES):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
        CATEGORIES (list): name of categories
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    if not isinstance(CATEGORIES[0], str):
        labels = [CATEGORIES[i]['name'] for i in labels]
    else:
        labels = [CATEGORIES[i] for i in labels]
    boxes = predictions.bbox
    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )
    return image

def vis_results(
        predictions, 
        img_infos,
        data_path,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224
    ):
    confidence_threshold = cfg.TEST.VIS_THRES
    mask_threshold = -1 if show_mask_heatmaps else 0.5
    masker = Masker(threshold=mask_threshold, padding=1)
    
    for prediction, img_info in zip(predictions, img_infos):
        img_name = img_info['file_name']
        image = cv2.imread(os.path.join(data_path, img_name))
        width, height = img_info['width'], img_info['height']
        assert image.shape[0] == height and image.shape[1] == width
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            masks = masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
            
        # select only prediction which have a `score` > confidence_threshold
        scores = prediction.get_field("scores")
        keep = torch.nonzero(scores > confidence_threshold, as_tuple=False).squeeze(1)
        prediction = prediction[keep]
        # prediction in descending order of score
        scores = prediction.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        prediction = prediction[idx]
        
        result = image.copy()
        if show_mask_heatmaps:
            result = create_mask_montage(result, prediction, masks_per_dim)
        else:
            result = overlay_boxes(result, prediction)    
            if cfg.MODEL.MASK_ON:
                result = overlay_mask(result, prediction)
            if cfg.MODEL.KEYPOINT_ON:
                result = overlay_keypoints(result, prediction)
            # name
            if 'coco' in data_path:
                CATEGORIES = COCO_CATEGORIES
            elif 'voc' in data_path:
                CATEGORIES = VOC_CATEGORIES
            else:
                raise ValueError
            result = overlay_class_names(result, prediction, CATEGORIES)
            
        # save
        out_path = os.path.join(cfg['OUTPUT_DIR'], 'vis')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path, img_name.replace('/', '-')), result)
        
def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
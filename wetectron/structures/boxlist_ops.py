# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList

from wetectron.layers import nms as _box_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)

# def boxlist_soft_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'):
#     """
#     Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503.

#     Arguments:
#         boxlist(BoxList)
#         nms_thresh (float)
#         max_proposals (int): if > 0, then only the top max_proposals are kept
#             after non-maximum suppression
#         score_field (str)
#     """
#     if nms_thresh <= 0:
#         return boxlist
#     mode = boxlist.mode
#     boxlist = boxlist.convert("xyxy")
#     boxes = boxlist.bbox
#     score = boxlist.get_field(score_field)
#     methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
#     assert method in methods, 'Unknown soft_nms method: {}'.format(method)
#     # TODO    
#     dets, keep = cython_nms.soft_nms(
#         np.ascontiguousarray(dets, dtype=np.float32),
#         np.float32(sigma),
#         np.float32(overlap_thresh),
#         np.float32(score_thresh),
#         np.uint8(methods[method])
#     )
    
#     if max_proposals > 0:
#         keep = keep[: max_proposals]
#     boxlist = boxlist[keep]
#     return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    # keep = (
    #     (ws >= min_size) & (hs >= min_size)
    # ).nonzero().squeeze(1)
    ## https://github.com/pytorch/vision/pull/2314
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.stack(torch.where(keep > 0), dim=1).squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou 

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou_async(boxlist1, boxlist2):
    """Compute the intersection over the area of boxlist1.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + 1e-10)
    return iou 

def batch_boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two batch set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [C,N,4].
      box2: (BoxList) bounding boxes, sized [C,M,4].

    Returns:
      (tensor) iou, sized [C,N,M].

    Reference: https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    assert boxlist1.size == boxlist2.size, "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2)
    assert boxlist1.bbox.shape[0] == boxlist2.bbox.shape[0]
    assert boxlist1.mode == "xyxy"
    assert boxlist2.mode == "xyxy"

    area1 = boxlist1.area() # [C, N]
    area2 = boxlist2.area() # [C, M]

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, :, None, :2], box2[:, None, :, :2])  
    rb = torch.min(box1[:, :, None, 2:], box2[:, None, :, 2:])  

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [C,N,M,2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [C,N,M]

    iou = inter / (area1[:, :, None] + area2[:, None, :] - inter)
    return iou

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes

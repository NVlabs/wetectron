# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from wetectron.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        if len(transposed_batch) == 3:
            img_ids = transposed_batch[2]
            return images, targets, img_ids
        elif len(transposed_batch) == 4:
            rois = transposed_batch[2]
            img_ids = transposed_batch[3]
            return images, targets, rois, img_ids
        else:
            raise ValueError('wrong item')


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))


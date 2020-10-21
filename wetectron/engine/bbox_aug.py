import torch
import torchvision.transforms as TT

from wetectron.config import cfg
from wetectron.data import transforms as T
from wetectron.structures.image_list import to_image_list
from wetectron.structures.bounding_box import BoxList
from wetectron.modeling.roi_heads.box_head.inference import make_roi_box_post_processor


def im_detect_bbox_aug(model, images, device, rois=None):
    # Collect detections computed under different transformations
    boxlists_ts = []
    for _ in range(len(images)):
        boxlists_ts.append([])

    def add_preds_t(boxlists_t):
        for i, boxlist_t in enumerate(boxlists_t):
            if len(boxlists_ts[i]) == 0:
                # The first one is identity transform, no need to resize the boxlist
                boxlists_ts[i].append(boxlist_t)
            else:
                # Resize the boxlist as the first one
                boxlists_ts[i].append(boxlist_t.resize(boxlists_ts[i][0].size))

    # Compute detections for the original image (identity transform)
    boxlists_i = im_detect_bbox(
        model, images, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, device, rois=rois
    )
    add_preds_t(boxlists_i)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        boxlists_hf = im_detect_bbox_hflip(
            model, images, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, device, rois=rois
        )
        add_preds_t(boxlists_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        boxlists_scl = im_detect_bbox_scale(
            model, images, scale, max_size, device, rois=rois
        )
        add_preds_t(boxlists_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            boxlists_scl_hf = im_detect_bbox_scale(
                model, images, scale, max_size, device, hflip=True, rois=rois
            )
            add_preds_t(boxlists_scl_hf)

    # Merge boxlists detected by different bbox aug params
    boxlists = []
    for i, boxlist_ts in enumerate(boxlists_ts):
        if cfg.TEST.BBOX_AUG.HEUR == 'UNION':
            bbox = torch.cat([boxlist_t.bbox for boxlist_t in boxlist_ts])
            scores = torch.cat([boxlist_t.get_field('scores') for boxlist_t in boxlist_ts])
        elif cfg.TEST.BBOX_AUG.HEUR == 'AVG':
            bbox = torch.mean(torch.stack([boxlist_t.bbox for boxlist_t in boxlist_ts]) ,  dim=0)
            scores = torch.mean(torch.stack([boxlist_t.get_field('scores') for boxlist_t in boxlist_ts]), dim=0)
        else:
            raise ValueError('please use proper BBOX_AUG.HEUR ')
        boxlist = BoxList(bbox, boxlist_ts[0].size, boxlist_ts[0].mode)
        boxlist.add_field('scores', scores)
        boxlists.append(boxlist)

    # Apply NMS and limit the final detections
    results = []
    post_processor = make_roi_box_post_processor(cfg)
    for boxlist in boxlists:
        results.append(post_processor.filter_results(boxlist, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES))

    return results


def im_detect_bbox(model, images, target_scale, target_max_size, device, rois=None):
    """
    Performs bbox detection on the original image.
    """    
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255
        )
    ])
    
    t_images = []
    t_rois = []
    for image, roi in zip(images, rois):  
        t_img, _, t_roi = transform(image, rois=roi)
        t_images.append(t_img)
        t_rois.append(t_roi)
    t_images = to_image_list(t_images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    t_rois = [r.to(device) if r is not None else None for r in t_rois]
    return model(t_images.to(device), rois=t_rois)


def im_detect_bbox_hflip(model, images, target_scale, target_max_size, device, rois=None):
    """
    Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """    
    transform = T.Compose([
        T.Resize(target_scale, target_max_size),
        T.RandomHorizontalFlip(1.0),
        T.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255
        )
    ])

    t_images = []
    t_rois = []
    
    for image, roi in zip(images, rois):  
        t_img, _, t_roi = transform(image, rois=roi)
        t_images.append(t_img)
        t_rois.append(t_roi)
    t_images = to_image_list(t_images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    t_rois = [r.to(device) if r is not None else None for r in t_rois]
    boxlists = model(t_images.to(device), rois=t_rois)

    # Invert the detections computed on the flipped image
    boxlists_inv = [boxlist.transpose(0) for boxlist in boxlists]
    return boxlists_inv


def im_detect_bbox_scale(model, images, target_scale, target_max_size, device, hflip=False, rois=None):
    """
    Computes bbox detections at the given scale.
    Returns predictions in the scaled image space.
    """
    if hflip:
        boxlists_scl = im_detect_bbox_hflip(model, images, target_scale, target_max_size, device, rois=rois)
    else:
        boxlists_scl = im_detect_bbox(model, images, target_scale, target_max_size, device, rois=rois)
    return boxlists_scl

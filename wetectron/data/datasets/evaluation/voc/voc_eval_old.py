# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from wetectron.structures.bounding_box import BoxList
from wetectron.structures.boxlist_ops import boxlist_iou
from wetectron.data.datasets.voc import PascalVOCDataset
from tqdm import tqdm
from six.moves import cPickle
import xml.etree.ElementTree as ET

def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap

def do_voc_evaluation(dataset, predictions, output_folder, logger):
    split = dataset.ann_file
    if "test" in dataset.ann_file:
        dataset = PascalVOCDataset("datasets/voc/VOC2012", "test")
    elif "val" in dataset.ann_file:
        dataset = PascalVOCDataset("datasets/voc/VOC2012", "val")

    class_boxes = {dataset.map_class_id_to_class_name(i + 1): [] for i in range(20)}
    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_bbox = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        pred_score = prediction.get_field("scores").numpy()

        for i, class_id in enumerate(pred_label):
            image_name = dataset.get_origin_id(image_id)
            box = pred_bbox[i]
            score = pred_score[i]
            class_name = dataset.map_class_id_to_class_name(class_id)
            class_boxes[class_name].append((image_name, box[0], box[1], box[2], box[3], score))
    aps = []
    tmp = os.path.join(output_folder, 'tmp')
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    for key in dataset.CLASSES[1:]:
        filename = os.path.join(output_folder, 'comp4_det_{}_{}.txt'.format(dataset.image_set, key))
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'wt') as txt:
            boxes = class_boxes[key]
            for k in range(len(boxes)):
                box = boxes[k]
                txt.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(box[0], box[-1], box[1], box[2], box[3], box[4]))
        annopath = os.path.join(dataset.root, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(dataset.root, 'ImageSets', 'Main', dataset.image_set+'.txt')
        if "val" in split:
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, key, tmp, ovthresh=0.5, use_07_metric=True)
            aps += [ap]
            logger.info('AP for {} = {:.4f}'.format(key, ap))
    if "val" in split:
        logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
        logger.info('~~~~~~~~')


count = 0
def dis_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5):
    """corloc = dis_eval(detpath,
                        annopath,
                        imagesetfile,
                        classname,
                        [ovthresh])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    imageset = os.path.splitext(os.path.basename(imagesetfile))[0]
    cachefile = os.path.join(cachedir, imageset + '_loc_annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = cPickle.load(f)

    class_recs = {}
    nimgs = 0.0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        nimgs = nimgs + float(bbox.size > 0)
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            global count
            count += 1

        if ovmax > ovthresh:
            tp[d] = 1.
            continue

    # print(classname, count, nimgs, nd)
    return np.sum(tp) / nimgs

def do_loc_evaluation(dataset, predictions, output_folder, logger):
    class_boxes = {dataset.map_class_id_to_class_name(i + 1): [] for i in range(20)}
    for image_id, prediction in tqdm(enumerate(predictions)):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_bbox = prediction.bbox.numpy()
        pred_label = prediction.get_field("labels").numpy()
        pred_score = prediction.get_field("scores").numpy()

        for i, class_id in enumerate(pred_label):
            image_name = dataset.get_origin_id(image_id)
            box = pred_bbox[i]
            score = pred_score[i]
            class_name = dataset.map_class_id_to_class_name(class_id)
            class_boxes[class_name].append((image_name, box[0], box[1], box[2], box[3], score))
    corlocs = []
    tmp = os.path.join(output_folder, 'tmp')
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    for key in dataset.CLASSES[1:]:
        filename = os.path.join(output_folder, '{}_loc.txt'.format(key))
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'wt') as txt:
            boxes = class_boxes[key]
            result = {}
            for k in range(len(boxes)):
                box = boxes[k]
                name = box[0]
                if name not in result.keys():
                    result[name] = box
                else:
                    if box[-1] > result[name][-1]:
                        result[name] = box
            for _, box in result.items():
                txt.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(box[0], box[-1], box[1], box[2], box[3], box[4]))
        annopath = os.path.join(dataset.root, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(dataset.root, 'ImageSets', 'Main', dataset.image_set+'.txt')
        corloc = dis_eval(filename, annopath, imagesetfile, key, tmp, ovthresh=0.5)
        corlocs += [corloc]
        logger.info('CorLoc for {} = {:.4f}'.format(key, corloc))
    logger.info('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
    logger.info('~~~~~~~~')
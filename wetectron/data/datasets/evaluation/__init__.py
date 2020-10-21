from wetectron.data import datasets
from .coco import coco_evaluation
from .voc import voc_evaluation


def evaluate(dataset, predictions, output_folder, task='det', **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset) and "voc_2012" not in dataset.ann_file:
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset) or "voc_2012" in dataset.ann_file:
        args['task'] = task
        return voc_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))

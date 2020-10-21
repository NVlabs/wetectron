import logging

from .voc_eval import do_voc_evaluation
from .voc_eval_old import do_loc_evaluation
from .voc_eval_old import do_voc_evaluation as do_voc_evaluation_old

def voc_evaluation(dataset, predictions, output_folder, box_only, task='det', **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    if task == 'det':
        return do_voc_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger,
        )
    if task == 'det_old':
        return do_voc_evaluation_old(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger,
        )
    elif task == 'corloc':
        return do_loc_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger,
        )
    else:
        raise ValueError

import logging

from .isic_eval import do_isic_evaluation
# from .isic_eval_noground import do_isic_evaluation

def isic_evaluation(dataset, predictions, grounds, output_folder, box_only, meters, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("isic evaluation doesn't support box_only, ignored.")
    logger.info("performing isic evaluation, ignored iou_types.")
    return do_isic_evaluation(
        dataset=dataset,
        predictions=predictions,
        grounds=grounds,
        output_folder=output_folder,
        logger=logger,
        meters=meters,
    )

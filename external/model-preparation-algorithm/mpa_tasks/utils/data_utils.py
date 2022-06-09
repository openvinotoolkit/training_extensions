
from mpa.utils.logger import get_logger

logger = get_logger()

def get_cls_img_indices(labels, dataset):
    #logger.info(labels)
    img_indices = {label.name: list() for label in labels}
    #logger.info(img_indices)
    for i, item in enumerate(dataset):
        item_labels = item.annotation_scene.get_labels()
        for i_l in item_labels:
            img_indices[i_l.name].append(i)
    logger.info(img_indices)
    return img_indices
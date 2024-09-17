import os
import os.path as osp
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.plugins.splitter import Split
from datumaro.plugins.data_formats.common_semantic_segmentation import (
    CommonSemanticSegmentationPath,
)
from datumaro.components.exporter import Exporter
from matplotlib.text import Annotation
from otx.core.utils.mask_util import polygon_to_bitmap
from datumaro.components.media import Image
from datumaro.components.annotation import (
    AnnotationType,
    LabelCategories,
    MaskCategories,
)
from datumaro.components.errors import MediaTypeError
from datumaro.util.definitions import DEFAULT_SUBSET_NAME
from datumaro.util.image import save_image
from datumaro.util.meta_file_util import save_meta_file
import numpy as np
from datumaro import Polygon, Bbox


class CommonSemanticSegmentationExporter(Exporter):
    """Exporter for common semantic segmentation format."""
    DEFAULT_IMAGE_EXT = ".jpg"

    def _apply_impl(self):
        """Apply the exporter to the dataset."""
        extractor = self._extractor
        save_dir = self._save_dir

        if self._extractor.media_type() and not issubclass(
            self._extractor.media_type(), Image
        ):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(save_dir, exist_ok=True)

        categories = extractor.categories()
        label_names = [label.name for label in categories[AnnotationType.label]]
        if "background" not in label_names:
            label_names.insert(0, "background")

        mask_categories = MaskCategories.generate(size=len(label_names))
        label_categories = LabelCategories()
        for label in label_names:
            label_categories.add(label)

        category = {
            AnnotationType.mask: mask_categories,
            AnnotationType.label: label_categories,
        }
        subsets = self._extractor.subsets()
        for subset_name, subset in subsets.items():
            if not subset_name or subset_name == DEFAULT_SUBSET_NAME:
                subset_name = DEFAULT_SUBSET_NAME

            subset_dir = osp.join(save_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            mask_dir = osp.join(subset_dir, CommonSemanticSegmentationPath.MASKS_DIR)
            img_dir = osp.join(subset_dir, CommonSemanticSegmentationPath.IMAGES_DIR)

            for item in subset:
                self._export_item_annotation(item, mask_dir)
                if self._save_media:
                    self._export_media(item, img_dir)
                save_meta_file(subset_dir, category)

    def _export_item_annotation(
        self,
        item: DatasetItem,
        save_dir: str,
    ):
        """Export the annotations of an item."""
        annotations = item.annotations
        height, width, _ = item.media.data.shape
        if not annotations:
            return

        index_map = np.zeros((height, width), dtype=np.uint8)
        for ann in annotations:
            if ann.type is AnnotationType.polygon:
                bitmask = polygon_to_bitmap([ann], height, width)[0]
                index_map[bitmask] = ann.label + 1
            elif ann.type is AnnotationType.ellipse:
                bitmask = polygon_to_bitmap([Polygon(ann.as_polygon(20))], height, width)[0]
                index_map[bitmask] = ann.label + 1
            elif ann.type is AnnotationType.bbox:
                x1, y1, w, h = [int(v) for v in ann.get_bbox()]
                index_map[y1: y1+h, x1: x1+w] = ann.label + 1
            else:
                raise NotImplementedError(
                    "Exporting %s is not supported" % ann.type
                )

        # standardise item id
        item_id = item.id.replace("/", "_")
        dst = osp.join(save_dir, item_id + ".png")

        # save index map as an image
        save_image(
            dst=dst,
            image=index_map,
            create_dir=True,
        )

    def _export_media(self, item, save_dir):
        """Export the media of an item."""
        item_id = item.id.replace("/", "_")
        dst = osp.join(save_dir, item_id + self.DEFAULT_IMAGE_EXT)
        save_image(
            dst=dst,
            image=item.media.data,
            create_dir=True,
        )
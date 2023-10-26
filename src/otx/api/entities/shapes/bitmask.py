import datetime
from typing import Optional

from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.shapes.shape import Shape, ShapeType
from otx.api.utils.time_utils import now
import numpy as np


class BitmapMask(Shape):
    def __init__(
        self,
        mask,
        x1,
        y1,
        x2,
        y2,
        modification_date: Optional[datetime.datetime] = None,
    ) -> None:
        modification_date = now() if modification_date is None else modification_date
        super().__init__(
            shape_type=ShapeType.BITMASK,
            modification_date=modification_date,
        )
        self.mask = mask
        self.box = np.array([x1, y1, x2 + 1, y2 + 1], dtype=np.int32)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __repr__(self):
        """String representation of the bitmap mask."""
        return f"Bitmap, min_x={self.x1}, min_y={self.y1}, max_x={self.x2}, max_y={self.y2})"

    def __hash__(self):
        return hash(str(self))

    @property
    def height(self) -> float:
        return self.mask.shape[0]

    @property
    def width(self) -> float:
        return self.mask.shape[1]

    def get_area(self) -> float:
        return self.mask.sum()

    def normalize_wrt_roi_shape(self, roi_shape: Rectangle):
        if not isinstance(roi_shape, Rectangle):
            raise ValueError("roi_shape has to be a Rectangle.")

        roi_shape = roi_shape.clip_to_visible_region()
        return self

    def normalize_wrt_roi(self, roi_shape: Rectangle) -> "BitmapMask":
        """The inverse of denormalize_wrt_roi_shape.

        Transforming Polygon from the `roi` coordinate system to the normalized coordinate system.
        This is used when the tasks want to save the analysis results.

        For example in Detection -> Segmentation pipeline, the analysis results of segmentation
        needs to be normalized to the roi (bounding boxes) coming from the detection.

        Args:
            roi_shape (Point): the shape of the roi
        """
        return self

    def denormalize_wrt_roi_shape(self, roi_shape: Rectangle) -> "BitmapMask":
        """The inverse of normalize_wrt_roi_shape.

        Transforming Polygon from the normalized coordinate system to the `roi` coordinate system.
        This is used to pull ground truth during training process of the tasks.
        Examples given in the Shape implementations.

        Args:
            roi_shape (Rectangle): the shape of the roi
        """
        return self

    def _as_shapely_polygon(self):
        return self

    def get_bbox(self):
        return self.box

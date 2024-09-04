import numpy as np
from model_api.models.utils import ImageResultWithSoftPrediction
from model_api.tilers import Tiler


class SegTiler(Tiler):
    def _postprocess_tile(
        self,
        predictions: ImageResultWithSoftPrediction,
        coord: list[int],
    ) -> dict:
        output_dict = {}
        output_dict["coord"] = coord
        output_dict["masks"] = predictions.soft_prediction
        return output_dict

    def _merge_results(self, results, shape):
        height, width = shape[:2]
        num_classes = len(self.model.labels)
        full_soft_mask = np.zeros((height, width, num_classes), dtype=np.float32)
        vote_mask = np.zeros((height, width), dtype=np.int32)
        for result in results:
            x1, y1, x2, y2 = result["coord"]
            mask = result["masks"]
            vote_mask[y1:y2, x1:x2] += 1
            full_soft_mask[y1:y2, x1:x2, :] += mask[: y2 - y1, : x2 - x1, :]

        # TODO: check correctness of this code
        full_soft_mask = full_soft_mask / vote_mask[:, :, None]
        index_mask = full_soft_mask.argmax(2)
        return ImageResultWithSoftPrediction(
            resultImage=index_mask,
            soft_prediction=full_soft_mask,
            feature_vector=np.array([]),
            saliency_map=np.array([]),
        )

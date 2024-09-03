from model_api.tilers import Tiler


# TODO
class SegTiler(Tiler):
    def _postprocess_tile(self, predictions, coord):
        return super()._postprocess_tile(predictions, coord)

    def _merge_results(self, results, shape):
        return super()._merge_results(results, shape)

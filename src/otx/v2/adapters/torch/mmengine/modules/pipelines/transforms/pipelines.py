import otx.v2.adapters.datumaro.pipelines.load_image_from_otx_dataset as load_image_base

from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""

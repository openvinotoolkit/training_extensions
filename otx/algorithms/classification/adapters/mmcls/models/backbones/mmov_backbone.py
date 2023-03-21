"""Module for the MMOVBackbone class."""

from typing import Dict, List

from mmcls.models.builder import BACKBONES

from otx.mpa.modules.ov.graph.parsers.cls.cls_base_parser import cls_base_parser
from otx.mpa.modules.ov.models.mmov_model import MMOVModel


@BACKBONES.register_module()
class MMOVBackbone(MMOVModel):
    """MMOVBackbone class.

    Args:
        *args: positional arguments.
        **kwargs: keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def parser(graph, **kwargs) -> Dict[str, List[str]]:
        """Parses the input and output of the model.

        Args:
            graph: input graph.
            **kwargs: keyword arguments.

        Returns:
            Dictionary containing input and output of the model.
        """
        output = cls_base_parser(graph, "backbone")
        if output is None:
            raise ValueError("Parser can not determine input and output of model. Please provide them explicitly")
        return output

    def init_weights(self, pretrained=None):  # pylint: disable=unused-argument
        """Initializes the weights of the model.

        Args:
            pretrained: pretrained weights. Default: None.
        """
        # TODO
        return

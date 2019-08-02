from .version import __version__
from .quantization import Quantization, Quantize, QuantizedNetwork, quantize
from .sparsity import ConstSparsity, ConstSparsifyingWeight
from .sparsity import MagnitudeSparsity, MagnitudeSparsifyingWeight
from .sparsity import RBSparsity, RBSparsifyingWeight

__all__ = [
    'QuantizedNetwork', 'Quantization', 'Quantize', 'quantize',
    'RBSparsity', 'RBSparsifyingWeight',
    'MagnitudeSparsity', 'MagnitudeSparsifyingWeight',
    'ConstSparsity', 'ConstSparsifyingWeight'

]

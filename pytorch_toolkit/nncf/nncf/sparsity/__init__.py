from .const import ConstSparsity
from .magnitude import MagnitudeSparsity
from .rb import RBSparsity, RBSparsifyingWeight

__all__ = [
    'RBSparsity', 'RBSparsifyingWeight',
    'MagnitudeSparsity', 'ConstSparsity'
]

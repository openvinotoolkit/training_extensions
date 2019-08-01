from .const import ConstSparsity, ConstSparsifyingWeight
from .magnitude import MagnitudeSparsity, MagnitudeSparsifyingWeight
from .rb import RBSparsity, RBSparsifyingWeight

__all__ = [
    'RBSparsity', 'RBSparsifyingWeight',
    'MagnitudeSparsity', 'MagnitudeSparsifyingWeight',
    'ConstSparsity', 'ConstSparsifyingWeight',
]

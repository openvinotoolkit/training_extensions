MMPRETRAIN_AVAILABLE = True

try:
    import mmpretrain  # noqa: F401
except ImportError:
    MMPRETRAIN_AVAILABLE = False

from .dataset import Dataset
from .engine import MMPTEngine as Engine
from .model import get_model

__all__ = ["get_model", "Dataset", "Engine"]

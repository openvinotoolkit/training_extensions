MMPRETRAIN_AVAILABLE = True

try:
    import mmpretrain  # noqa: F401
    from mmpretrain.utils import register_all_modules

    register_all_modules(init_default_scope=True)
except ImportError:
    MMPRETRAIN_AVAILABLE = False

from .dataset import Dataset
from .engine import MMPTEngine as Engine
from .model import get_model, list_models

__all__ = ["get_model", "Dataset", "Engine", "list_models"]

AVAILABLE = True
VERSION = None

try:
    import torch  # noqa: F401

    VERSION = torch.__version__
except ImportError:
    AVAILABLE = False

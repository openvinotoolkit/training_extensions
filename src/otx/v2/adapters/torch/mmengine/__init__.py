AVAILABLE = True
VERSION = None

try:
    import mmengine  # noqa: F401

    VERSION = mmengine.__version__
except ImportError:
    AVAILABLE = False

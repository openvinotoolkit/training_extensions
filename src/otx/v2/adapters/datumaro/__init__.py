AVAILABLE = True
VERSION = None

try:
    import datumaro  # noqa: F401

    VERSION = datumaro.__version__
except ImportError:
    AVAILABLE = False

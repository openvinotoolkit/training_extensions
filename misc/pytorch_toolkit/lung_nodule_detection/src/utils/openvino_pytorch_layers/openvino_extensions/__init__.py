import os
import sys

def get_extensions_path():
    lib_name = 'user_cpu_extension'
    if sys.platform == 'win32':
        lib_name += '.dll'
    elif sys.platform == 'linux':
        lib_name = 'lib' + lib_name + '.so'
    else:
        lib_name = 'lib' + lib_name + '.dylib'
    return os.path.join(os.path.dirname(__file__), lib_name)


# This is a dummy procedure which instantiates onnx_importer library preloading
try:
    import io
    from openvino.inference_engine import IECore
    ie = IECore()
    buf = io.BytesIO()
    ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
except Exception:
    pass

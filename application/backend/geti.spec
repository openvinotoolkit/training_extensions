# -*- mode: python ; coding: utf-8 -*-
import glob
import platform
import os
import yaml
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_submodules, collect_data_files, copy_metadata

# When set to "1", exclude Ultralytics (AGPL-3.0) models from the bundle.
EXCLUDE_AGPL_MODELS = os.environ.get("EXCLUDE_AGPL_MODELS", "0") == "1"

_AGPL_MODULES = ['ultralytics', 'thop', 'pynvml']

_MANIFESTS_ROOT = 'app/supported_models/manifests'

def _is_agpl_manifest(path):
    """Return True if a manifest YAML declares an AGPL-3.0 license.

    Fails closed: if the manifest cannot be read or parsed, it is treated as
    AGPL so it is excluded rather than accidentally shipped when the
    compliance-driven EXCLUDE_AGPL_MODELS flag is enabled.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return True
    if not isinstance(manifest, dict):
        return True
    return manifest.get('license') == 'AGPL-3.0'

def _collect_manifests(exclude_agpl):
    """Collect model manifests (recursively), preserving subdirectory layout.

    When exclude_agpl is True, AGPL-3.0 licensed manifests are dropped to mirror
    the manifest stripping done in docker/Dockerfile.
    """
    result = []
    for path in glob.glob(os.path.join(_MANIFESTS_ROOT, '**', '*'), recursive=True):
        if not os.path.isfile(path):
            continue
        if exclude_agpl and path.endswith(('.yaml', '.yml')) and _is_agpl_manifest(path):
            continue  # skip AGPL-licensed model manifest
        # Preserve the manifest's subdirectory structure in the bundle.
        dest_dir = os.path.dirname(path)
        result.append((path, dest_dir))
    return result

datas = [
    ('app/alembic', 'app/alembic'),
    ('app/alembic.ini', 'app'),
    ('app/static/*', 'app/static'),
    # SAM encoder model shipped inside the app/services/sam module
    ('app/services/sam/mobile_sam.encoder.xml', 'app/services/sam'),
    ('app/services/sam/mobile_sam.encoder.bin', 'app/services/sam'),
    # timm model catalog snapshot, read via importlib.resources at runtime
    ('app/supported_models/timm_catalog_snapshot.json', 'app/supported_models'),
    *_collect_manifests(EXCLUDE_AGPL_MODELS),
    *copy_metadata("geti"),
    *copy_metadata("optree"),
    *copy_metadata("torch"),
    *copy_metadata("tabulate"),
    *copy_metadata("matplotlib"),
    *copy_metadata("lightning"),
    *copy_metadata("torchmetrics"),
    *copy_metadata("jsonargparse"),
    *copy_metadata("rich"),
]
binaries = [(dll, 'Library/bin/') for dll in glob.glob('.venv/Library/bin/*')]
hiddenimports = []

# ---- PyTorch core ----
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('torch.backends')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('triton')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Lightning (training framework) ----
tmp_ret = collect_all('lightning')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('lightning.pytorch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Model training dependencies ----
tmp_ret = collect_all('torchmetrics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('timm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('kornia')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('jsonargparse')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('einops')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('omegaconf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Model export dependencies ----
tmp_ret = collect_all('onnx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('onnxscript')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('onnxconverter_common')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Inference / quantization dependencies ----
tmp_ret = collect_all('openvino')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('model_api')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('nncf')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Logging / metrics dependencies ----
tmp_ret = collect_all('tensorboard')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('tensorboardX')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('faster_coco_eval')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- ML / data dependencies ----
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# scipy (pulled in by sklearn); collect_all to capture dynamically-imported
# submodules such as scipy._external.array_api_compat.numpy.fft
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
hiddenimports += collect_submodules('scipy._external.array_api_compat')

tmp_ret = collect_all('polars')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- getitune and model packages ----
tmp_ret = collect_all('getitune')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('rfdetr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('datumaro')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('pytorchcv')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- OpenCV (camera capture backends) ----
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- WebRTC dependencies ----
tmp_ret = collect_all('aiortc')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('aioice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('pylibsrtp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('av')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('ifaddr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all('google_crc32c')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ---- Explicit hidden imports for spawned-process training ----
# When PyInstaller runs a frozen app that spawns child processes (multiprocessing "spawn"),
# these modules must be discoverable by the child even though they are only imported
# dynamically inside the training / export code paths.
hiddenimports += [
    # Lightning internals used by Trainer
    'lightning.pytorch.callbacks',
    'lightning.pytorch.loggers',
    'lightning.pytorch.plugins',
    'lightning.pytorch.strategies',
    'lightning.pytorch.profilers',
    'lightning.pytorch.accelerators',
    # PyTorch submodules needed for model export
    'torch.export',
    'torch.onnx',
    'torch.onnx.symbolic_opset10',
    'torch.onnx.symbolic_opset11',
    'torch.optim',
    'torch.optim.lr_scheduler',
    'torch.distributed',
    # jsonargparse internals used for model instantiation
    'jsonargparse._actions',
    'jsonargparse._typehints',
    'jsonargparse._loaders_dumpers',
    'jsonargparse._parameter_resolvers',
    'jsonargparse._link_arguments',
    'jsonargparse._optionals',
    'jsonargparse._util',
    'jsonargparse._common',
    'jsonargparse._namespace',
    'jsonargparse._signatures',
    # scikit-learn (used by SSD model for KMeans anchors)
    'sklearn.cluster',
    'sklearn.utils',
    # Multiprocessing support in frozen applications
    'multiprocessing.spawn',
    'multiprocessing.popen_spawn_win32',
    'multiprocessing.popen_spawn_posix',
    'multiprocessing.resource_tracker',
]

# Ensure cryptography is bundled so the runtime cert-generation hook (windows/certs.py)
# can create a self-signed TLS certificate on first launch.
hiddenimports += collect_submodules('cryptography')

# Runtime hook to patch importlib.metadata must execute before torch is imported
# in every process (including multiprocessing-spawned children).
runtime_hooks = ['pyinstaller/pyi_rth_pkgmeta.py']

system = platform.system()
if system == "Windows":
    # uwp.py sets DATA_DIR; certs.py must run after it to generate TLS certs there.
    runtime_hooks += ['pyinstaller/windows/uwp.py', 'pyinstaller/windows/certs.py', 'pyinstaller/windows/proxy.py']

a = Analysis(
    ['app/main.py'],
    pathex=['app'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[
        'torch.utils.benchmark',
        *(_AGPL_MODULES if EXCLUDE_AGPL_MODELS else []),
    ],
    noarchive=False,
    optimize=0,
)

def _is_excluded(name, excluded_patterns):
    normalized = name.replace('\\', '/')
    return any(pattern in normalized for pattern in excluded_patterns)

# Filter out redundant triton backends (nvidia, amd) not needed for XPU distribution
_excluded_triton_backends = ('triton/backends/nvidia', 'triton/backends/amd')
a.binaries = [b for b in a.binaries if not _is_excluded(b[0], _excluded_triton_backends)]
a.datas = [d for d in a.datas if not _is_excluded(d[0], _excluded_triton_backends)]

# Remove non-redistributable DLLs from MSIX distribution
_excluded_dlls = ('torch/lib/cusolverMg64_11.dll', 'torch/lib/cusolverMg64_12.dll', 'torch/lib/nvperf_host.dll')
a.binaries = [b for b in a.binaries if not _is_excluded(b[0], _excluded_dlls)]

# When excluding AGPL models, drop any data/binaries transitively collected from
# the AGPL packages uninstalled in docker/Dockerfile (ultralytics,
# ultralytics-thop -> thop, nvidia-ml-py -> pynvml/nvidia_ml_py).
if EXCLUDE_AGPL_MODELS:
    _excluded_agpl = ('ultralytics', 'thop', 'pynvml', 'nvidia_ml_py')
    a.binaries = [b for b in a.binaries if not _is_excluded(b[0], _excluded_agpl)]
    a.datas = [d for d in a.datas if not _is_excluded(d[0], _excluded_agpl)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='geti-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='geti-backend',
)

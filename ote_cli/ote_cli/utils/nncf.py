import torch

def is_checkpoint_nncf(path):
    """
    The function uses metadata stored in a checkpoint to check if the
    checkpoint was the result of trainning of NNCF-compressed model.
    """
    try:
        state = torch.load(path, map_location='cpu')
        return bool(state.get('meta',{}).get('nncf_enable_compression', False))
    except FileNotFoundError:
        return False

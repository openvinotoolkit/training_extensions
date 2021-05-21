import json
from addict import Dict

def load_config(cfg_path):
    cfg = Dict(json.load(open(cfg_path)))
    return cfg

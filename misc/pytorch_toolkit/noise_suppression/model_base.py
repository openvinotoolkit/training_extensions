import logging
import os
import json


import torch
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} model_base'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))


class BaseDNSModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.init_kwargs = kwargs
        printlog("Create {} model with params:".format(type(self).__name__))
        for k,v in kwargs.items():
            printlog(k,v)

    #save load functions
    WEIGHTS_NAME = "model.pth"
    CONFIG_NAME = "config.json"
    def save_pretrained(self, dir_name):
        weights_name = os.path.join(dir_name, self.WEIGHTS_NAME)
        config_name = os.path.join(dir_name, self.CONFIG_NAME)
        torch.save(self.state_dict(), weights_name)
        config_dict = {
            'class_name': type(self).__name__,
            'init_kwargs': self.init_kwargs
        }
        with open(config_name,'wt') as f:
            json.dump(config_dict, f)

    @classmethod
    def from_config(cls, config_name, MODELS=None):
        with open(config_name,'rt') as f:
            config_dict = json.load(f)

        class_name = config_dict['class_name']
        init_kwargs = config_dict['init_kwargs']
        if 'model_desc' in config_dict:
            model_desc = config_dict['model_desc']
            init_kwargs['model_desc'] = model_desc

        if MODELS:
            assert cls is BaseDNSModel
            if class_name in MODELS:
                model = MODELS[class_name](**init_kwargs)
            else:
                raise RuntimeError("{} class is not found in {}".format(class_name, list(MODELS.keys())))
        else:
            model = cls(**init_kwargs )
        return model

    @classmethod
    def from_pretrained(cls, dir_name):
        weights_name = os.path.join(dir_name, cls.WEIGHTS_NAME)
        config_name = os.path.join(dir_name, cls.CONFIG_NAME)
        model = cls.from_config(config_name)
        state_dict = torch.load(weights_name, map_location="cpu")
        model.load_state_dict(state_dict)
        return model


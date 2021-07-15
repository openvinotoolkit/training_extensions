import logging
import os
import json
from model_base import BaseDNSModel
from model_poconetlike import PoCoNetLikeModel

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} models'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

#MODELS = {cls.__name__:cls for cls in globals().values() if isinstance(cls, type) and issubclass(cls, BaseDNSModel)}
MODELS = {
    'PoCoNetLikeModel': PoCoNetLikeModel
}
#check that model class based on  BaseDNSModel
assert all(issubclass(cls, BaseDNSModel) for cls in MODELS.values())

def model_from_dir(model_dir):

    config_name = os.path.join(model_dir, BaseDNSModel.CONFIG_NAME)
    with open(config_name, 'rt') as f:
        config_dict = json.load(f)
    class_name = config_dict["class_name"]
    printlog("load {} class from {}".format(class_name, model_dir))
    if class_name in MODELS:
        model = MODELS[class_name].from_pretrained(model_dir)
    else:
        raise RuntimeError("{} class is not found in {}".format(class_name, list(MODELS.keys())))
    return model

def model_create(model_desc):
    #load or create model
    if os.path.isdir(model_desc):
        #model from directory
        model = model_from_dir(model_desc)
    elif os.path.isfile(model_desc):
        #model from config
        model = BaseDNSModel.from_config(model_desc, MODELS=MODELS)
    else:
        assert "\\" not in model_desc, "probably model description is bad dir or bad file"
        assert "/" not in model_desc, "probably model description is bad dir or bad file"
        #model from description. from class name for example
        names = [n for n in MODELS.keys() if n in model_desc]
        if names:
            names.sort(key=len, reverse=True) #take the longest match
            printlog("Create not initialized model {} from list of similar models {}".format(names[0], names))
            model = MODELS[names[0]](model_desc=model_desc)
        else:
            msg = "{} model description does not match with any class in {}".format(model_desc, list(MODELS.keys()))
            raise RuntimeError(msg)

    return model

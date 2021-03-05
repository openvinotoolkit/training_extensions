from collections import namedtuple

from torch import nn

from . import resnet
from . import mobilenetv2
from . import rmnet

Encoder = namedtuple('Encoder', ('model', 'features', 'features_shape'))


def make_encoder(name, input_size=224, input_channels=3, pretrained=None):
    """Make encoder (backbone) with a given name and parameters"""
    
    features_size = input_size // 32
    num_features = 2048
    if name.startswith('resnet'):
        model = getattr(resnet, name)(pretrained=pretrained, num_channels=input_channels)
        features = nn.Sequential(*list(model.children())[:-2])
        num_features = 512 if int(name[6:]) < 50 else 2048
    elif name.startswith('mobilenetv2'):
        model = mobilenetv2.MobileNetV2(input_size=input_size, pretrained=None)
        features = model.features
        num_features = 1280
    elif name.startswith('rmnet'):
        model = rmnet.RMNetClassifier(1000, pretrained=None)
        features = nn.Sequential(*list(model.children())[:-2])
        num_features = 512
    elif name.startswith('se_res'):
        model = load_from_pretrainedmodels(name)(pretrained='imagenet' if pretrained else None)
        features = nn.Sequential(*list(model.children())[:-2])
    else:
        raise KeyError("Unknown model name: {}".format(name))

    features_shape = (num_features, features_size, features_size)
    return Encoder(model, features, features_shape)


def load_from_pretrainedmodels(model_name):
    import pretrainedmodels
    return getattr(pretrainedmodels, model_name)

from __future__ import absolute_import

from .TCLNet import TCLNet

__factory = {
        'TCLNet': TCLNet,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)

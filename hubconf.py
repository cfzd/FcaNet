dependencies = ['torch']
from model.fcanet import fcanet34 as _fcanet34
from model.fcanet import fcanet50 as _fcanet50
from model.fcanet import fcanet101 as _fcanet101
from model.fcanet import fcanet152 as _fcanet152
import torch
# entrypoints:
def fca34(pretrained=False):
    model = _fcanet34(pretrained=pretrained)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url('https://github.com/cfzd/FcaNet/releases/download/v1.0/fca34.pth')
        model.load_state_dict(state_dict)
    return model

def fca50(pretrained=False):
    model = _fcanet50(pretrained=pretrained)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url('https://github.com/cfzd/FcaNet/releases/download/v1.0/fca50.pth')
        model.load_state_dict(state_dict)
    return model

def fca101(pretrained=False):
    model = _fcanet101(pretrained=pretrained)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url('https://github.com/cfzd/FcaNet/releases/download/v1.0/fca101.pth')
        model.load_state_dict(state_dict)
    return model

def fca152(pretrained=False):
    model = _fcanet152(pretrained=pretrained)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url('https://github.com/cfzd/FcaNet/releases/download/v1.0/fca152.pth')
        model.load_state_dict(state_dict)
    return model
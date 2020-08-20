import torch.nn as nn
from models import models

def freezeResNet(model):
    for name, p in model.named_parameters():
        if 'fc' not in name:
            p.required_grad = False
    return model #이거 필요한가?

def freezeDenseNet(model):
    for name, p in model.named_parameters():
        if 'classifier' not in name:
            p.required_grad = False
    return model

def freezeVGG(model):
    for name, p in model.named_parameters():
        if 'classifier.6' not in name:
            p.required_grad = False
    return model


def fineTuningModel(name, num_classes, is_freeze=True, pretrained=False): #freeze

    model = models(name, pretrained) 
    if 'resnet' in name:
        input_features = model.fc.in_features
        model.fc = nn.Linear(in_features = input_features, out_features = num_classes, bias=True)
        if is_freeze and pretrained :
            model = freezeResNet(model)
    elif 'vgg' in name:
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features = input_features, out_features = num_classes, bias=True)
        if is_freeze and pretrained:
            model = freezeVGG(model)
    elif 'densenet' in name:
        input_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features = input_features, out_features = num_classes, bias=True)
        if is_freeze and pretrained:
            model = freezeDenseNet(model)
    
    return model
    

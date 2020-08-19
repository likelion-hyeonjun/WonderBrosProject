import torchvision.models as models

def load(Name, pretrained=False):
    if Name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=pretrained)
    elif Name == 'densenet169':
        model = torchvision.models.densenet169(pretrained=pretrained)
    elif Name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=pretrained)
    elif Name == 'densenet201':
        model = torchvision.models.densenet201(pretrained=pretrained)
    elif Name == 'vgg11':
        model = torchvision.models.vgg11(pretrained=pretrained)
    elif Name == 'vgg11_bn':
        model = torchvision.models.vgg11_bn(pretrained=pretrained)
    elif Name == 'vgg13':
        model = torchvision.models.vgg13(pretrained=pretrained)
    elif Name == 'vgg16' :
        model = torchvision.models.vgg16(pretrained=pretrained)
    elif Name == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
    elif Name == 'vgg19':
        model = torchvision.models.vgg19(pretrained=pretrained)
    elif Name == 'vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained=pretrained)
    elif Name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
    elif Name == 'resnet34' :
        model = torchvision.models.resnet34(pretrained=pretrained)
    elif Name == 'resnet50' :
        model = torchvision.models.resnet50(pretrained=pretrained)
    elif Name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
    elif Name == 'resnet152':
        model = torchvision.models.resnet152(pretrained=pretrained)
    else :
        model = torchvision.models.vgg19(pretrained=pretrained) #그냥 이거 디폴트함

    return model



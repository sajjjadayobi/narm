import torch.nn as nn
from torchvision import models
import torchvision.models._utils as _utils

class ResnetCNN(nn.Module):

    '''
    the separator of CNN on Resnet networks
    this Works with all Resent family (resnet, resnext, wide_resnet)

    name: name of resnet family networks 
    pretrained: pre-trained weights for transfer learning
    to layer: all of Resnet networks have 4 convolutional block 
              and 4 meaning all of the convolutional block
    '''
    def __init__(self, name='resnet50', pretrained=False, to_layer=4):
        super(ResnetCNN, self).__init__()

        resnet_family = models.__dict__[name](pretrained=pretrained)
        self.cnn = _utils.IntermediateLayerGetter(resnet_family, {f'layer{to_layer}': 0})

    def forward(self, x):
        return self.cnn(x)

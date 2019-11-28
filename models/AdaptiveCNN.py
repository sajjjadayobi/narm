import torch
import torch.nn as nn
from torchvision import models

class AdaptiveCNN(nn.Module):

    """ adaptive convolutional neural network
            input: 
              tensor: image with any size
              int: feature_map_size
            output:
              a tensor with size (n_batch, 512, feature_map_size, feature_map_size)
            
            point 1: we used pre-trained Resnet18 for CNN layer
            point 2: 512 is the size of the channel in the final conv layer

        global average pooling(GAP) in paper:
            Fully connected layers are prone to overfitting, this is obviously less prone
            Saves a lot of parameters
    """

    def __init__(self, feature_map_size=1):
        super(AdaptiveCNN, self).__init__()
        model = models.resnet18(pretrained=True)


        self.cnn = nn.Sequential(
                      model.conv1,
                      model.bn1,
                      model.relu,
                      model.maxpool,
                      model.layer1,
                      model.layer2,
                      model.layer3,
                      model.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(feature_map_size, feature_map_size))
        self.clf = nn.Softmax2d()
    

    def forward(self, x):
        x = self.cnn(x)
        x = self.avgpool(x)
        x = self.clf(x)
        return x

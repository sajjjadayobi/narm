import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------
# Freez CNN and then fine-tune

class Net(nn.Module):
  def __init__(self, number_of_classes):
    super(Net, self).__init__()
    self.model = models.resnet50()
    # use resnet50 CNN & own FC
    self.model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, number_of_classes),)

  def forward(self, x):
    return self.model(x)


# define random data
torch.random.seed = 0
random_input = torch.randn(2, 3, 64, 64)
random_target = torch.tensor([0, 1]).long()

net = Net(2)
# just add fc.param for update
# with this trick, we can learn any layers which defined as a specific layer
optimizer = optim.Adam(net.model.fc.parameters(), lr=0.0005)
# also with follwing code you can Frezz any layer you want
''' for p in net.model.fc.parameters():
        p.requires_grad = False '''
    
criterion = nn.CrossEntropyLoss()
# trainig section
for i in range(3):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    optimizer.step()
    print(loss.data)

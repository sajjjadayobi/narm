import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


# toy feed-forward net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# define random data
random_input = Variable(torch.randn(10,))
random_target = Variable(torch.randn(1,))

net = Net()

# print the pre-trained fc2 weight
print('fc2 pretrained weight (same as the one above):')
print(net.fc2.weight)

# define new random data
random_input = Variable(torch.randn(10,))
random_target = Variable(torch.randn(1,))

# we want to freeze the fc2 layer this time: only train fc1 and fc3
for p in net.fc2.parameters():
    p.requires_grad = False

# train again
criterion = nn.MSELoss()

# NOTE: pytorch optimizer explicitly accepts parameter that requires grad
# see https://github.com/pytorch/pytorch/issues/679
optimizer = optim.Adam(net.parameters(), lr=0.1)
# this raises ValueError: optimizing a parameter that doesn't require gradients
#optimizer = optim.Adam(net.parameters(), lr=0.1)

for i in range(100):
    net.zero_grad()
    output = net(random_input)
    loss = criterion(output, random_target)
    loss.backward()
    optimizer.step()

# print the retrained fc2 weight
# note that the weight is same as the one before retraining: only fc1 & fc3 changed
print('fc2 weight (frozen) after retrain:')
print(net.fc2.weight)

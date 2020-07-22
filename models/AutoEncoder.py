# Mnist reconstructed example with Conv AE

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import mnist
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms

# download & extract Mnist 
train_ds = mnist.MNIST('train', train=True, download=True)
test_ds = mnist.MNIST('test', train=False, download=True)
x_train, y_train = train_ds.data, np.array(train_ds.targets)
x_test, y_test = test_ds.data, np.array(train_ds.targets)
print('train set: ', x_train.shape)
print('test set: ', x_test.shape)

# create train & test loder 
train_ds.transform = transforms.Compose([transforms.ToTensor()])
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, num_workers=8, shuffle=True)
test_ds.transform = transforms.Compose([transforms.ToTensor(),])
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, num_workers=4, shuffle=True)

# show some of these data
x, y = next(iter(train_dl))
indexs = np.random.choice(y, replace=False, size=10)
num_classes = train_ds.classes

fig, axis = plt.subplots(2, 5, figsize=(25, 8))
for i, ax in zip(indexs, axis.flat):
    ax.set_title(num_classes[y[i]])
    ax.imshow(x[i][0], cmap='gray')
    ax.axis('off')
    
    

# create AutoEncoder Model
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
             nn.Conv2d(1, 16, kernel_size=3, padding=(1, 1)),
             nn.BatchNorm2d(16),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=(2, 2)),
             nn.Conv2d(16, 8, kernel_size=3, padding=(1, 1)),
             nn.BatchNorm2d(8),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=(2, 2)),
             nn.Conv2d(8, 8, kernel_size=3, padding=(1, 1)),
             nn.BatchNorm2d(8),
             nn.ReLU(),)

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(size=(14, 114)),
            nn.Conv2d(8, 16, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(size=(28, 28)),
            nn.Conv2d(16, 1, kernel_size=3, padding=(1, 1)),  
            nn.Sigmoid(),)

    def forward(self, x):
        x = self.encoder(x)
        self.code = x
        x = self.decoder(x)
        return  x
    

# Create a Training Class
class TrainAE():

    def __init__(self, model, train_dl, optimizer, criterion, num_epochs=1, scheduler=None, valid_dl=None, cuda=False):
        self.num_epochs = num_epochs
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cuda = cuda
        if cuda:
        self.model = model.cuda()
        self.criterion = criterion.cuda()
        else:
        self.model = model
        self.criterion = criterion


        self.valid_loss_history = []
        self.train_loss_history = []
        self.best_loss_valid = 0.0
        self.best_wieght = None
        self.training()

    def training(self):

        valid_acc = 0
        for epoch in range(self.num_epochs):

            print('Epoch %2d/%2d'%(epoch + 1, self.num_epochs))
            print('-' * 12)

            t0 = time.time()
            train_loss = self.train_model()
            if self.valid_dl:
            valid_loss = self.valid_model()
            if self.scheduler:
            self.scheduler.step()

            time_elapsed = time.time() - t0
            print('\nEpochTime: %.0fm %.0fs | valid_loss: %.3f | train_loss: %.3f\n'%(time_elapsed//60, time_elapsed%60, valid_loss, train_loss))

            if valid_loss < self.best_loss_valid:
                self.best_loss_valid = valid_loss
                self.best_wieght = self.model.state_dict().copy() 
        return
  

    def train_model(self):

        self.model.train()
        N = len(self.train_dl.dataset)
        step = N // self.train_dl.batch_size
        avg_loss = torch.tensor(0.0, requires_grad=False)

        for i, (x, y) in enumerate(self.train_dl):
            if self.cuda:
                x = x.cuda()
            # forward
            scores = self.model(x)
            # loss
            loss =  self.criterion(scores, x)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # statistics of model training
            avg_loss  = (avg_loss * i + loss) / (i + 1)
            self.train_loss_history.append(avg_loss)

            # report statistics
            sys.stdout.flush()
            sys.stdout.write("\r  Train_Step: %d/%d | runing_loss: %.4f"%(i+1, step, avg_loss))

            sys.stdout.flush()
        return avg_loss


    def valid_model(self):
        with torch.no_grad():
            self.model.eval()
            N = len(self.valid_dl.dataset)
            step =  N // self.valid_dl.batch_size
            avg_loss = torch.tensor(0.0, requires_grad=False)      

            for i, (x, y) in enumerate(self.valid_dl):
                if self.cuda:
                    x = x.cuda()
                # forward
                scores = self.model(x)
                # loss
                loss =  self.criterion(scores, x)
                # statistics of model training
                avg_loss  = (avg_loss * i + loss) / (i + 1)
                self.valid_loss_history.append(avg_loss)

                sys.stdout.flush()
                sys.stdout.write("\r  Vaild_Step: %d/%d | runing_loss: %.4f" % (i, step, avg_loss))

            sys.stdout.flush()
        return avg_loss

    
    
# create a learning process
model = ConvAutoEncoder()
optimizer = optim.Adam(lr=0.01, params=model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=.5)
criterion = nn.BCELoss() # or nn.MSELoss()
History = train_model(model=model, train_dl=train_dl, valid_dl=test_dl, optimizer=optimizer, criterion=criterion, scheduler=scheduler,  num_epochs=1)


# row 1 show original data
# row 2 show reconstructed data
x, y = next(iter(test_dl))
indexs = np.random.choice(y, replace=False, size=5)
num_classes = train_ds.classes

fig, axis = plt.subplots(1, 5, figsize=(25, 8))
for i, ax in zip(indexs, axis.flat):
    ax.imshow(x[i][0], cmap='gray')
    ax.axis('off')

model.eval()
X = model(x).cpu().detach().numpy()

fig, axis = plt.subplots(1, 5, figsize=(25, 8))
for i, ax in zip(indexs, axis.flat):
    ax.imshow(X[i][0], cmap='gray_r')
    ax.axis('off')

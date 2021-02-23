import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


# trainers
from trainer import Trainer
from callbacks import Reporter, LRFinder

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (255))])
train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# write  your custom metrics
def acc(pred, y):
    pred = torch.argmax(pred, dim=1)
    return torch.sum(pred == y)


model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 128), nn.GELU(), nn.Linear(128, 10))
loss = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

learner = Trainer(model=model, train_ds=train_ds, valid_ds=test_ds, optimizer=optimizer, loss=loss, metrcis=[acc])
learner.summary()


lr_finder = LRFinder(min_lr=1e-5, max_lr=1e1, epoch=1)
learner.fit(epochs=2, device='cpu', callbacks=[Reporter(), lr_finder])

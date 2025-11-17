
import torch
import torch.nn as nn
class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN,self).__init__()
    self.n_classes=n_classes
    self.n_channels=n_channels
    self.sequential=nn.Sequential(
        nn.Conv2d(in_channels=n_channels,out_channels=64,kernel_size=3,padding=1,stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1,stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        nn.Flatten(),
        nn.Linear(in_features=512,out_features=10),
    )
  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    out=self.sequential(x)
    return out

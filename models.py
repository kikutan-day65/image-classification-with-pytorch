import torch
import torch.nn as nn

class ImageNetCNN(nn.Module):
    def __init__(self, num_class):
        super(ImageNetCNN, self).__init__()

        # convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        
        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully-connected layer
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
    
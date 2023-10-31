import torch.nn as nn


class CNNImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=64*37*37, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=6, bias=True)
        )
    
    def forward(self, x):
        return self.network(x)
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


data_dir = './datasets'

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

dataset = ImageFolder(data_dir, transform=transform)

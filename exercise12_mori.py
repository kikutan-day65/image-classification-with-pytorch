import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


data_dir = './datasets'

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

dataset = ImageFolder(data_dir, transform=transform)

batch_size = 64

dataset_size = len(dataset)

train_ratio = 0.8
val_ratio = 0.2

train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size

train_data, val_data = random_split(dataset, [train_size, val_size])

train_dl = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_data, batch_size*2, num_workers=4, pin_memory=True)

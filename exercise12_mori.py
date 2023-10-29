import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader

data_dir = './datasets'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=data_dir, transform=transform)

dataset_size = len(dataset)

train_ratio = 0.8
test_ratio = 0.2

train_size = int(train_ratio * dataset_size)
test_size = int(test_ratio * dataset_size)
val_size = dataset_size - train_size - test_size

train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

print(f"トレーニングセットのサイズ: {len(train_dataset)}")
print(f"テストセットのサイズ: {len(test_dataset)}")
print(f"バリデーションセットのサイズ: {len(val_dataset)}")

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from cnnModel import NaturalSceneClassification
from baseModel import fit


def main():
    data_dir = './datasets'

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(data_dir, transform=transform)

    dataset_size = len(dataset)

    # 80% for train data, otherwise test data
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    # randomly split the data
    train_data, test_data = random_split(dataset, [train_size, test_size])

    batch_size = 64

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

   
if __name__ == '__main__':
    main()
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from exercise12_2_mori import CNNImageClassifier


def main():
    data_dir = './datasets'

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
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

    # just make sure using device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # create model instance
    model = CNNImageClassifier()

    # loss functtion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    ## training section
    num_epochs = 50
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # split to the mini batch
        for batch in train_dl:

            # extract corresponding image data and labels
            images, labels = batch

            # reset gradient to zero (required before loop new mini batch)
            optimizer.zero_grad()

            # get prediction
            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, labels)

            # back propagation
            loss.backward()

            # update weight
            optimizer.step()

            # add each loss to total loss
            total_loss += loss.item()

        # print losses each loop
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dl)}")

    ## evaluation section
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_dl:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {100 * accuracy:.2f}%")

    # save the trained model
    torch.save(model.state_dict(), 'trained_model.pth')


if __name__ == '__main__':
    main()
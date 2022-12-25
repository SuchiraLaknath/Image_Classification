import sys

import torch
from torch import nn
import torchvision.models as models
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision import transforms
import torchvision
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/resnet50/adam")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Resnet50(nn.Module):
    def __init__(self, num_classes=5):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, 5)

    def forward(self, x):
        return (self.model(x))


def check_validation(test_loader, model, epoch):
    num_correct = 0
    running_loss = 0.0
    total = 0
    model.eval()

    with torch.no_grad():
        for data, targets in tqdm(test_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)
            _, predictions = torch.max(scores, 1)
            running_loss += loss
            num_correct += (predictions == targets).sum().item()
            total += targets.size(0)
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.00 * num_correct / total

    writer.add_scalar('Testing loss', epoch_loss, epoch)
    writer.add_scalar('Testing accuracy', epoch_acc, epoch)
    print("Testing dataset. got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" % (
        num_correct, total, epoch_acc, epoch_loss))


def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        print("Epoch number ", (epoch + 1))
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # print("Current batch : ", batch_idx)
            data = data.to(device=device)
            targets = targets.to(device=device)
            total += targets.size(0)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            _, predicted = torch.max(scores.data, 1)

            # backward
            optimizer.zero_grad()
            loss.backward()
            # gradient descent or adam step
            optimizer.step()
            running_loss += loss.item()
            running_correct += (targets == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.00 * running_correct / total

        writer.add_scalar('Training loss', epoch_loss, epoch)
        writer.add_scalar('Training accuracy', epoch_acc, epoch)
        print(" Training dataset. got %d out of %d images correctly (%.3f%%). Epoch loss: %.3f" % (
            running_correct, total, epoch_acc, epoch_loss))
        print("validation on epoch ", (epoch + 1), " : ")
        check_validation(test_loader, model, epoch)

    print("Training Finished")

    return model


train_dataset_path = './data/train'
test_dataset_path = './data/test'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    # transforms.Grayscale(1),

    transforms.Resize(64),
    transforms.GaussianBlur(3),
    # transforms.AutoAugmentPolicy(0),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    #transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transforms = transforms.Compose([
    # transforms.Grayscale(1),
    transforms.Resize(64),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    #transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_dataset = ImageFolder(root=train_dataset_path, transform=train_transforms)
test_dataset = ImageFolder(root=test_dataset_path, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

examples = iter(train_loader)
examples_data, example_targets = examples.next()

img_grid = torchvision.utils.make_grid(examples_data)
writer.add_image('Train_images', img_grid)

model = Resnet50(num_classes=5).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model = train_nn(model, train_loader, test_loader, criterion, optimizer, 20)
print("Final Validation")

writer.add_graph(model, examples_data.to(device))

# check_validation(test_loader,model)

torch.save(model.state_dict(), 'model_adam.pkl')

writer.close()
sys.exit()

import torch
from torch import nn
import torch.nn.modules as Model
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
from torch import  nn

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from torchvision import transforms

import numpy as np


class Resnet50(nn.Module):
    def __init__(self, num_classes=5):
        super(Resnet50, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(2048, 5)

    def forward(self, x):
        return (self.model(x))

model = Resnet50()
model.load_state_dict(torch.load('model_adam.pkl'))
model.eval()



train_dataset_path = './data/train'
test_dataset_path = './data/test'


mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]

train_transforms = transforms.Compose([
    #transforms.Grayscale(1),

    transforms.Resize(64),
    transforms.GaussianBlur(3),
    #transforms.AutoAugmentPolicy(0),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    #transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

test_transforms = transforms.Compose([
    #transforms.Grayscale(1),
    transforms.Resize(64),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    #transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])


#train_dataset = ImageFolder(root = train_dataset_path, transform = train_transforms)
test_dataset = ImageFolder(root = test_dataset_path, transform = test_transforms)


#train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=10,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=10,shuffle=True)


classes = ['1 _Lifeboy', '2 _Drink', '3 _Marmite', '4 _Buiscuit', '5 _charger']


from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

for inputs, labels in test_loader:
    output = model(inputs)  # Feed Network

    print(labels)

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output)  # Save Prediction

    labels = labels.data.cpu().numpy()
    y_true.extend(labels)  # Save Truth

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix , index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')



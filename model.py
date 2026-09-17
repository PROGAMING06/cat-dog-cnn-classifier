import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            32 * 16 * 16,
            64
        )

        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(
            64,
            2
        )

    def forward(self, x):
        x = self.pool1(
            self.relu1(
                self.conv1(x)
            )
        )

        x = self.pool2(
            self.relu2(
                self.conv2(x)
            )
        )

        x = self.flatten(x)

        x = self.relu3(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.fc2(x)

        return x
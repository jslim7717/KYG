import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=12, out_channels=100, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=250, kernel_size=10)
        self.conv3 = nn.Conv1d(in_channels=250, out_channels=500, kernel_size=10)
        self.conv4 = nn.Conv1d(in_channels=500, out_channels=1000, kernel_size=10)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(250)
        self.bn3 = nn.BatchNorm1d(500)
        self.bn4 = nn.BatchNorm1d(1000)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear(1000, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)

        x = self.fc(x)
        x = self.sigmoid(x)
        return x

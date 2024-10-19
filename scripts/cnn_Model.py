import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScreenshotToneClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ScreenshotToneClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x shape: (batch_size, 3, 224, 224)

        # Pass through convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # (batch_size, 128, 28, 28)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 28 * 28)  # Flatten to (batch_size, 128 * 28 * 28)

        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here, as this will be handled by the loss function

        return x
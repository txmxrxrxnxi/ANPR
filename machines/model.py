import torch
import torch.nn as nn

class ANPRModel(nn.Module):
    """
    Convolutional Neural Network model for Automatic Number Plate Recognition (ANPR).

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        pool (nn.MaxPool2d): Max pooling layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(self):
        """
        Initializes the ANPR model with convolutional and fully connected layers.
        """
        
        super(ANPRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with bounding box predictions.
        """
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 26 * 26)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

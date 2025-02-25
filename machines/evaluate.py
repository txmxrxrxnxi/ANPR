import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torch import nn


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: torch.device):
    """
    Evaluates the ANPR model on the test data.

    Args:
        model (nn.Module): The ANPR model to evaluate.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate the model on.
    """

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, bboxes in test_loader:
            inputs, bboxes = inputs.to(device), bboxes.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, bboxes)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader)}")

def plot_predictions(images: np.ndarray, bboxes: np.ndarray, predictions: torch.Tensor, index: int):
    """
    Plots the predictions of the ANPR model.

    Args:
        images (np.ndarray): Array of images.
        bboxes (np.ndarray): Array of true bounding boxes.
        predictions (torch.Tensor): Array of predicted bounding boxes.
        index (int): Index of the image to plot.
    """

    img_tensor = torch.tensor(images[index], dtype=torch.float32).permute(2, 0, 1)
    bbox_true = bboxes[index]
    bbox_pred = predictions[index].cpu().numpy()

    img = img_tensor.permute(1, 2, 0).cpu().numpy()

    plt.imshow(img)
    plt.gca().add_patch(plt.Rectangle((bbox_true[0], bbox_true[1]), bbox_true[2], bbox_true[3], 
                                      edgecolor='green', facecolor='none', linewidth=2, label='True'))
    plt.gca().add_patch(plt.Rectangle((bbox_pred[0], bbox_pred[1]), bbox_pred[2], bbox_pred[3], 
                                      edgecolor='red', facecolor='none', linewidth=2, label='Predicted'))
    plt.legend()
    plt.show()

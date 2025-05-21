import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import torch
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

def plot_predictions(rows, folder_path, model, target_size, device):
    """
    Plots six random images with real and predicted bounding boxes in a single matplotlib window.

    Args:
        rows (list): List of rows containing annotations.
        folder_path (str): Path to the folder containing images.
        model: Trained model.
        target_size (tuple): Target size for resizing images.
        device: Device to perform computation on (e.g., 'cuda' or 'cpu').
    """
    # Choose six random annotations
    selected_annotations = random.sample(rows, 6)

    # Create a matplotlib figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns for 6 images
    axes = axes.ravel()  # Flatten axes for easy iteration

    model.eval()
    for i, annotation in enumerate(selected_annotations):
        img_path = os.path.join(folder_path, annotation[5])  # row[5] is the file_name
        real_bbox = [annotation[1], annotation[2], annotation[3], annotation[4]]  # bbox_x, bbox_y, bbox_width, bbox_height

        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load image:", img_path)
            continue
        
        # Resize and normalize image
        img_resized = cv2.resize(img, target_size)
        img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Predict bounding box
        with torch.no_grad():
            predicted_bbox = model(img_tensor).cpu().numpy().squeeze()

        # Plot image and bounding boxes
        ax = axes[i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        real_rect = patches.Rectangle((real_bbox[0], real_bbox[1]), real_bbox[2], real_bbox[3],
                                       linewidth=2, edgecolor='g', facecolor='none', label='Real BBox')
        predicted_rect = patches.Rectangle((predicted_bbox[0], predicted_bbox[1]), predicted_bbox[2], predicted_bbox[3],
                                            linewidth=2, edgecolor='r', facecolor='none', label='Predicted BBox')
        ax.add_patch(real_rect)
        ax.add_patch(predicted_rect)
        ax.set_title(f"Image {i + 1}")
        ax.axis('off')

    # Adjust layout and show the plot
    fig.tight_layout()
    plt.legend(['Real BBox', 'Predicted BBox'], loc="upper right")
    plt.show()
    
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import generalized_box_iou
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn

import pandas as pd
import matplotlib.pyplot as plt


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def calculate_accuracy(outputs, targets, threshold=0.5):
    predicted_boxes = outputs.cpu().detach().numpy()
    target_boxes = targets.cpu().detach().numpy()

    correct_predictions = 0
    for pred_box, target_box in zip(predicted_boxes, target_boxes):
        iou = calculate_iou(pred_box, target_box)
        if iou >= threshold:
            correct_predictions += 1

    accuracy = correct_predictions / len(target_boxes)
    return accuracy

def giou_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes GIoU loss = 1 - GIoU(preds, targets).
    preds and targets are (N, 4) in [x1, y1, x2, y2] format.
    """
    # generalized_box_iou returns a Tensor of shape (N,)
    giou = generalized_box_iou(preds, targets)
    return (1 - giou).mean()

def is_contained(pred_box, target_box, tol: float = 0.0) -> bool:
    """
    Returns True if target_box is fully inside pred_box, 
    within an optional tolerance in pixels.
    Boxes are in [x1, y1, x2, y2] format.
    """
    px1, py1, px2, py2 = pred_box
    tx1, ty1, tx2, ty2 = target_box

    return (
        px1 - tol <= tx1 and
        py1 - tol <= ty1 and
        px2 + tol >= tx2 and
        py2 + tol >= ty2
    )

def calculate_containment_accuracy(outputs, targets, tol: float = 0.0):
    """
    outputs: tensor of shape (batch, 4)
    targets: tensor of shape (batch, 4)
    tol: allowed slack (in pixels) on each side.
    """
    preds = outputs.detach().cpu().numpy()
    targs = targets.detach().cpu().numpy()
    count_inside = 0

    for p, t in zip(preds, targs):
        if is_contained(p, t, tol):
            count_inside += 1

    return count_inside / len(targs)

HISTORY_FILE = "history/training_history.csv"

def save_history(train_loss, valid_loss, train_acc, valid_acc):
    """Appends training history to a CSV file."""
    data = pd.DataFrame([[train_loss, valid_loss, train_acc, valid_acc]],
                        columns=["Train Loss", "Valid Loss", "Train Accuracy", "Valid Accuracy"])
    
    try:
        existing_data = pd.read_csv(HISTORY_FILE)
        data = pd.concat([existing_data, data], ignore_index=True)
    except FileNotFoundError:
        pass  # If file doesn't exist, start fresh

    data.to_csv(HISTORY_FILE, index=False)

def load_history():
    """Loads training history from CSV file."""
    try:
        return pd.read_csv(HISTORY_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Train Loss", "Valid Loss", "Train Accuracy", "Valid Accuracy"])

def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, 
                criterion: nn.Module, device: torch.device, 
                epochs: int = 10, show_plots: bool = False, model_name: str = "anpr_model.pth"):
    """
    Trains the ANPR model.

    Args:
        model (nn.Module): The ANPR model to train.
        train_loader (DataLoader): DataLoader for the training data.
        valid_loader (DataLoader): DataLoader for the validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train the model on.
        epochs (int): Number of training epochs.
        show_plots (bool): Whether to show the accuracy and loss history plots.
    """
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=1e-3, threshold_mode='abs')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0.0
        total_train = 0

        for inputs, bboxes in train_loader:
            inputs = inputs.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = giou_loss(outputs, bboxes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                giou_vals = generalized_box_iou(outputs, bboxes).diag()
                batch_iou_acc = (giou_vals >= 0.5).float().mean().item()
            correct_train += batch_iou_acc
            total_train += 1

        train_loss = running_loss / len(train_loader)
        train_acc  = correct_train / total_train

        model.eval()
        valid_loss_sum = 0.0
        correct_valid = 0.0
        total_valid = 0

        with torch.no_grad():
            for inputs, bboxes in valid_loader:
                inputs = inputs.to(device)
                bboxes = bboxes.to(device)
                outputs = model(inputs)

                loss = giou_loss(outputs, bboxes)
                valid_loss_sum += loss.item()

                giou_vals = generalized_box_iou(outputs, bboxes).diag()
                batch_iou_acc = (giou_vals >= 0.5).float().mean().item()
                correct_valid += batch_iou_acc
                total_valid += 1

        valid_loss = valid_loss_sum / len(valid_loader)
        valid_acc  = correct_valid / total_valid

        scheduler.step(valid_loss)

        save_history(train_loss, valid_loss, train_acc, valid_acc)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train Accuracy: {train_acc}, Valid Accuracy: {valid_acc}")

    torch.save(model.state_dict(), model_name)
    print("Model saved!")

    if show_plots:
        history = load_history()
        num_epochs = len(history["Train Accuracy"])
        epochs_range = list(range(1, num_epochs + 1))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history["Train Loss"], label='Train Loss')
        plt.plot(epochs_range, history["Valid Loss"], label='Valid Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()
        plt.savefig('history/loss_history.png')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history["Train Accuracy"], label='Train Accuracy')
        plt.plot(epochs_range, history["Valid Accuracy"], label='Valid Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy History')
        plt.legend()
        plt.savefig('history/accuracy_history.png')

        plt.show()

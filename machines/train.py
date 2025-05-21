import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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

HISTORY_FILE = "history/training_history.csv"

def save_history(epoch, train_loss, valid_loss, train_acc, valid_acc):
    """Appends training history to a CSV file."""
    data = pd.DataFrame([[epoch, train_loss, valid_loss, train_acc, valid_acc]],
                        columns=["Epoch", "Train Loss", "Valid Loss", "Train Accuracy", "Valid Accuracy"])
    
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
        return pd.DataFrame(columns=["Epoch", "Train Loss", "Valid Loss", "Train Accuracy", "Valid Accuracy"])

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, threshold=1e-3, threshold_mode='abs')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, bboxes in train_loader:
            inputs, bboxes = inputs.to(device), bboxes.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, bboxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            correct_train += calculate_accuracy(outputs, bboxes)
            total_train += 1

        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, bboxes in valid_loader:
                inputs, bboxes = inputs.to(device), bboxes.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, bboxes)
                valid_loss += loss.item()

                correct_valid += calculate_accuracy(outputs, bboxes)
                total_valid += 1

        train_loss = running_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        scheduler.step(valid_loss)
        
        train_acc = correct_train / total_train
        valid_acc = correct_valid / total_valid

        save_history(epoch, train_loss, valid_loss, train_acc, valid_acc)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train Accuracy: {train_acc}, Valid Accuracy: {valid_acc}")

    torch.save(model.state_dict(), model_name)
    print("Model saved!")

    if show_plots:
        history = load_history()

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history["Epoch"], history["Train Loss"], label='Train Loss')
        plt.plot(history["Epoch"], history["Valid Loss"], label='Valid Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()
        plt.savefig('history/loss_history.png')

        plt.subplot(1, 2, 2)
        plt.plot(history["Epoch"], history["Train Accuracy"], label='Train Accuracy')
        plt.plot(history["Epoch"], history["Valid Accuracy"], label='Valid Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy History')
        plt.legend()
        plt.savefig('history/accuracy_history.png')

        plt.show()

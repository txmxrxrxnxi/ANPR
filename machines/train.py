import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt


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
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, bboxes in train_loader:
            inputs, bboxes = inputs.to(device), bboxes.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, bboxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, bboxes in valid_loader:
                inputs, bboxes = inputs.to(device), bboxes.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, bboxes)
                valid_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Losses: {train_loss}, Valid Losses: {valid_loss}")

    torch.save(model.state_dict(), model_name)
    print("Model saved!")

    if show_plots:
        epochs_range = range(1, epochs + 1)
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, valid_losses, label='Valid Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()

        plt.show()

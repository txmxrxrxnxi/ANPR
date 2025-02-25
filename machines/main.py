import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ANPRDataset
from model import ANPRModel
from train import train_model
from evaluate import evaluate_model

TARGET_SIZE = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_dataset = ANPRDataset('data/train', TARGET_SIZE)
    test_dataset = ANPRDataset('data/test', TARGET_SIZE)
    valid_dataset = ANPRDataset('data/valid', TARGET_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    model = ANPRModel()
    criterion = nn.MSELoss()
    model.to(device)
    train_model(model, train_loader, valid_loader, criterion, device, epochs=10, show_plots=True)
    evaluate_model(model, test_loader, criterion, device)


if __name__ == "__main__":
    main()
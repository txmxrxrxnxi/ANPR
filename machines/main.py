import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
from dotenv import load_dotenv
import argparse

import psycopg2

from dataset import ANPRDataset
from model import ANPRModel
from train import train_model
from evaluate import evaluate_model, plot_predictions


load_dotenv(encoding="utf-8")
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 10

def arg_parse():
    parser = argparse.ArgumentParser(description="Train an ANPR model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--show_plots", action="store_true", help="Show accuracy and loss plots")

    return parser.parse_args()

def connect_db():
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"))

    return conn

def get_rows(conn):
    cur = conn.cursor()
    cur.execute("SELECT image_id, bbox_x, bbox_y, bbox_width, bbox_height, file_name FROM annotations_train")
    rows_train = cur.fetchall()
    cur.execute("SELECT image_id, bbox_x, bbox_y, bbox_width, bbox_height, file_name FROM annotations_test")
    rows_test = cur.fetchall()
    cur.execute("SELECT image_id, bbox_x, bbox_y, bbox_width, bbox_height, file_name FROM annotations_valid")
    rows_valid = cur.fetchall()
    conn.close()

    return rows_train, rows_test, rows_valid

def main():
    parser = arg_parse()

    conn = connect_db()
    rows_train, rows_test, rows_valid = get_rows(conn)

    train_dataset = ANPRDataset(rows_train, "data/train", TARGET_SIZE)
    test_dataset = ANPRDataset(rows_test, "data/test", TARGET_SIZE)
    valid_dataset = ANPRDataset(rows_valid, "data/valid", TARGET_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ANPRModel()
    criterion = nn.SmoothL1Loss()

    if parser.model:
        model.load_state_dict(torch.load(parser.model))

    epochs = int(parser.epochs) if parser.epochs else 10

    model.to(device)
    train_model(model, train_loader, valid_loader, criterion, device, epochs=epochs, show_plots=True)
    evaluate_model(model, test_loader, criterion, device)
    plot_predictions(rows_valid, "data/valid", model, TARGET_SIZE, device)

if __name__ == "__main__":
    main()
    
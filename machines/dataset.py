import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ANPRDataset(Dataset):
    """
    Custom Dataset for Automatic Number Plate Recognition (ANPR) that loads images and bounding boxes.

    Attributes:
        images (np.ndarray): Array of images.
        bboxes (np.ndarray): Array of bounding boxes corresponding to the images.
        target_size (tuple): Target size for resizing images.
    """

    def __init__(self, folder_path: str, target_size: tuple):
        """
        Initializes the dataset by loading images and bounding boxes.

        Args:
            folder_path (str): Path to the folder containing images and annotations JSON file.
            target_size (tuple): Target size for resizing images.
        """

        self.images, self.bboxes = self.load_data(folder_path)
        self.target_size = target_size

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """

        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Retrieves the image and bounding box at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Transformed image tensor and bounding box tensor.
        """

        img = cv2.resize(self.images[idx], self.target_size)
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        bbox = self.bboxes[idx]
        return img, torch.tensor(bbox, dtype=torch.float32)

    def load_data(self, folder_path: str) -> tuple:
        """
        Loads images and bounding boxes from the specified folder.

        Args:
            folder_path (str): Path to the folder containing images and annotations JSON file.

        Returns:
            tuple: Arrays of images and bounding boxes.
        """
        
        with open(folder_path + ".json") as f:
            data = json.load(f)

        images = []
        bboxes = []

        for annotation in data['annotations']:
            img_path = os.path.join(folder_path, annotation['file_name'])
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                bbox = annotation['bbox']
                if len(bbox) == 4:
                    bboxes.append(bbox)

        images = np.array(images, dtype=object)
        bboxes = np.array(bboxes, dtype=float)
        return images, bboxes

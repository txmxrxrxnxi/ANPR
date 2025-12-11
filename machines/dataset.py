import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

class ANPRDataset(Dataset):
    """
    Custom Dataset for Automatic Number Plate Recognition (ANPR) that loads images and bounding boxes.

    Attributes:
        images (np.ndarray): Array of images.
        bboxes (np.ndarray): Array of bounding boxes corresponding to the images.
        target_size (tuple): Target size for resizing images.
    """

    def __init__(self, rows, folder_path: str, target_size: tuple):
        """
        Initializes the dataset by loading images and bounding boxes.

        Args:
            folder_path (str): Path to the folder containing images and annotations JSON file.
            target_size (tuple): Target size for resizing images.
        """

        super().__init__()
        self.target_size = target_size
        self.images, self.bboxes = self.load_data(rows, folder_path)
        return

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

    def get_transforms(target_size):
        """
        Defines data transformations for preprocessing and augmentation.
        """

        return A.Compose([
                A.Affine(rotate=(-45, 45), translate_percent=0.0625, scale=(0.9, 1.1), p=0.7),
                A.RandomResizedCrop(size=(target_size[1], target_size[0]), scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.3),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def load_data(self, rows, folder_path: str) -> tuple:
        """
        Loads images and bounding boxes from the specified folder.

        Args:
            rows: 
            folder_path (str): Path to the folder containing images and annotations JSON file.

        Returns:
            tuple: Arrays of images and bounding boxes.
        """
        images = []
        bboxes = []

        tr = ANPRDataset.get_transforms(self.target_size)

        for row in rows:
            img_path = os.path.join(folder_path, row[5])
            img = cv2.imread(img_path)
            if img is not None:
                # img = tr(img)
                images.append(img)
                bbox = [row[1], row[2], row[3], row[4]]
                bboxes.append(bbox)

        images = np.array(images, dtype=object)
        bboxes = np.array(bboxes, dtype=float)
        return images, bboxes
    

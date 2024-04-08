import os
import cv2
import torch
import albumentations as A
import json
import re
from PIL import Image
from torchvision import transforms
import config as CFG
import pdb

class Dataset_for_query(torch.utils.data.Dataset):
    def __init__(self, df, image_path, transforms=None, args=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the image filenames and locations.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transforms = transforms
        self.args = args,
        self.image_path = image_path

    def __getitem__(self, idx):
        # Retrieve image filename and location from the DataFrame
        image_file = self.df.iloc[idx]['image']

        # Construct the full image path
        image_path = os.path.join(self.image_path, image_file)

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations if any
        if self.transforms:
            image = self.transforms(image=image)['image']

        # Convert image and location to tensors
        image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

        item = {'image': image_tensor}

        return item

    def __len__(self):
        return len(self.df)

class CPIPDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_path, transforms=None, args=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the image filenames and locations.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transforms = transforms
        self.args = args,
        self.image_path = image_path

    def __getitem__(self, idx):
        # Retrieve image filename and location from the DataFrame
        image_file = self.df.iloc[idx]['image']
        location = self.df.iloc[idx]['location']

        # Construct the full image path
        image_path = os.path.join(self.image_path, image_file)

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations if any
        if self.transforms:
            image = self.transforms(image=image)['image']

        # Convert image and location to tensors
        image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        location_tensor = torch.tensor(location, dtype=torch.float)

        item = {'image': image_tensor, 'location': location_tensor}

        return item

    def __len__(self):
        return len(self.df)

def get_transforms(mode="train", args=None):
        return A.Compose(
            [
                A.Resize(args.img_height, args.img_width, always_apply=True),
                A.Normalize(max_pixel_value=255.0, mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225], always_apply=True),
            ]
        )


    
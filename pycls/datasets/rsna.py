"""RSNA dataset."""

from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.utils.data

from pycls.core.config import cfg

import pycls.datasets.transforms as transforms
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class RSNA(torch.utils.data.Dataset):
    def __init__(self, data_path, csv_path, transforms = None):
        csv = pd.read_csv(csv_path)
        self.fnames = [os.path.join(data_path, patient_id + ".jpg")
                           for patient_id in csv['patientId']]   # get the full path to images
        self.labels = csv['Target'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.fnames) # size of dataset

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        img = Image.open(self.fnames[idx])
        if self.transforms != None:
            img = self.transforms(img) # Apply Specific Transformation to Image
        return img, self.labels[idx]

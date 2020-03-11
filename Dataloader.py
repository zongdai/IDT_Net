import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class IDT_Net_Dataset(Dataset):

    def __init__(self, root_dir, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ])):
        
       	self.source_images = os.listdir(os.path.join(root_dir, 'source'))
       	self.target_images = os.listdir(os.path.join(root_dir, 'target'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return min(len(os.listdir(os.path.join(self.root_dir, 'source'))), 
        	len(os.listdir(os.path.join(self.root_dir, 'target'))))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        source_image = io.imread(os.path.join(self.root_dir, 'source', self.source_images[idx]))
        target_image = io.imread(os.path.join(self.root_dir, 'target', self.target_images[idx]))
       	
        

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        sample = {
        	'source' : source_image,
        	'target' : target_image
        }
        return sample
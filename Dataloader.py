import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import cv2
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

    def get_patches(self, source, target):
        source_patches = []
        target_patches = []
        for i in range(5):
            for j in range(19):
                # print(source.shape)
                item = source[:, i*64:(i+1)*64, j*64:(j+1)*64]
                source_patches.append(item.unsqueeze(0))
        for i in range(161+256):
            x = random.randint(0, 1200 - 64 -1)
            y = random.randint(0, 350 - 64 -1)
            source_patches.append(source[:, y:y+64, x:x+64].unsqueeze(0))
        source_patches = torch.cat(source_patches)

        for i in range(512):
            x = random.randint(0, 1024 - 64 -1)
            y = random.randint(0, 512 - 64 -1)
            target_patches.append(target[:, y:y+64, x:x+64].unsqueeze(0))
        target_patches = torch.cat(target_patches)
        return source_patches, target_patches

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        source_image = cv2.imread(os.path.join(self.root_dir, 'source', self.source_images[idx]))
        target_image = cv2.imread(os.path.join(self.root_dir, 'target', self.target_images[idx]))
        source_image = source_image[...,[2,1,0]]
        target_image = target_image[...,[2,1,0]]
       	target_image = cv2.resize(target_image, (1024, 512))
        # target_image = transform.resize(target_image, (1024, 512), 3, anti_aliasing=False)
        # print(target_image.shape)

        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        source_patches, target_patches = self.get_patches(source_image, target_image)
        sample = {
        	'source' : source_patches,
        	'target' : target_patches
        }
        return sample
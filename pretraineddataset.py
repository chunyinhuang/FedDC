import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random







class PretrainedDataset(Dataset):
    """
    Code for reading the Pretrained dataset
    """

    def __init__(self, dataset_path='./pretrained', dataset='CovidX', ipc = 50, padding = 2, im_size = [224, 224]):

        self.root = str(dataset_path)
        self.dataset = dataset

        if self.dataset[-6:] == 'CovidX':
            self.classes = [0,1]
            mean = [0.4886, 0.4886, 0.4886]
            std = [0.2460, 0.2460, 0.2460]
        elif self.dataset[-8:] == 'ImageNet':
            self.classes = np.arange(1000)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise NotImplementedError
        
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std)
            ])
        
        img_path = os.path.join(dataset_path, dataset+'.png')
        images_pil = Image.open(img_path).convert('RGB')
        images_torch = transform(images_pil)
        self.images = []
        self.labels = []
        for j in range(len(self.classes)):
            for i in range(ipc):
                self.images.append(images_torch[:, (padding+im_size[0])*j+padding:(padding+im_size[0])*j+padding+im_size[0], (padding+im_size[1])*i+padding:(padding+im_size[1])*i+padding+im_size[1]])
                self.labels.append(j)

        print(f'{dataset}', images_torch.size(), len(self.images), len(self.labels), self.images[1].size())

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, index):
        return self.images[index] #, self.labels[index]

        

    def read_filepaths(self):

        paths, labels = [], []
        # print(file)
        splitpoint = 10000
        with open(self.file, 'r') as f:
            lines = f.read().splitlines()

            for index, line in enumerate(lines):

                if self.flag == 'train':
                    if index >= splitpoint:
                        break
                elif self.flag == 'test':
                    if index<splitpoint:
                        continue
                else:
                    raise NotImplementedError

                subjid, path, label = line.split(' ')[:3]

                paths.append(path)
                labels.append(label)
            

        return paths, labels





if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # from matplotlib import patches, patheffects
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image


    norm_mean = (0.4886, 0.4886, 0.4886)
    norm_std = (0.2460, 0.2460, 0.2460)
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop((224, 224)), transforms.ToTensor()])
    
   

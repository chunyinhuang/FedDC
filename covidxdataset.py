import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random







class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, dataset_path='../../MAE/data/covid', transform=None, flag='train'):

        self.root = str(dataset_path)
        self.flag = flag
        self.transform = transform

        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1}

        self.file = os.path.join(self.root, 'allsingle.txt')
        self.paths, self.labels = self.read_filepaths()
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image = self.load_image(os.path.join(self.root, 'COVID', self.paths[index]), index)

        image_tensor = self.transform(image)

        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path):

        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        
        return image

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
    
   

    # # check each mean and std
    # print('check each mean and std')
    
    # training_set = COVIDxDataset(transform=val_transform, train=False)
    # for i in range(len(training_set)):
    #     image, label = training_set[i]
    #     psum    = image.sum(axis        = [1, 2])
    #     psum_sq = (image ** 2).sum(axis = [1, 2])
    #     count = 224 * 224
    #     total_mean = psum / count
    #     total_var  = (psum_sq / count) - (total_mean ** 2)
    #     total_std  = torch.sqrt(total_var)
    #     print('mean: '  + str(total_mean) + ', std:  '  + str(total_std))


    # count class ratio
    print('count class ratio')
    training_set = COVIDxDataset('./data/covid', flag='test', blur=False, noisy=True)
    found0 = 0
    found1 = 0
    print(f'length of dataset:{len(training_set)}')
    # for i in range(len(training_set)):
    #     image, label = training_set[i]
    #     # print(label)
    #     # # block
    #     # image[0,40:160,30:194] = 1.0
    #     # image[1,40:160,30:194] = 1.0
    #     # image[2,40:160,30:194] = 1.0
    #     if label == 1:
    #         found1 += 1
    #     elif label == 0:
    #         found0 += 1
    # print(f'found {found0} 0s, and {found1} 1s.')


    # # save noisy images
    # print('whatsup')
    # training_set = COVIDxDataset('./data/covid', flag='value', blur=False, noisy=True)
    # idx = [1, 101, 201, 301, 401]
    # for i in idx:
    #     save_image(training_set[i][0], f'COVIDX_noisy_{i}.png')

    

    # found0 = False
    # found1 = False
    # for i in range(100):
    #     image, label = training_set[i]
    #     # # block
    #     # image[0,40:160,30:194] = 1.0
    #     # image[1,40:160,30:194] = 1.0
    #     # image[2,40:160,30:194] = 1.0
    #     if label == 1 and found1 is not True:
    #         print(f'Found 1 in {i}')
    #         save_image(image, f'COVIDX_1_noise0.png')
    #         found1 = True
    #     if label == 0 and found0 is not True:
    #         print(f'Found 0 in {i}')
    #         save_image(image, f'COVIDX_0_noise0.png')
    #         found0 = True
    #     if found0 is True and found1 is True:
    #         break


        


    # # find mean and std
    # training_set = COVIDxDataset(flag='value_clean')
    # image_loader = DataLoader(training_set, 
    #                       batch_size  = 256, 
    #                       shuffle     = False, 
    #                       num_workers = 2,
    #                       pin_memory  = True)

    # # placeholders
    # psum    = torch.tensor([0.0, 0.0, 0.0])
    # psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # # loop through images
    # for inputs, _ in tqdm(image_loader):
    #     psum    += inputs.sum(axis        = [0, 2, 3])
    #     psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])
    
    # # pixel count
    # count = len(training_set) * 224 * 224

    # # mean and std
    # total_mean = psum / count
    # total_var  = (psum_sq / count) - (total_mean ** 2)
    # total_std  = torch.sqrt(total_var)

    # # output
    # print('mean: '  + str(total_mean))
    # print('std:  '  + str(total_std))
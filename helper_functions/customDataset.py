import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import cv2
import numpy as np

class LungXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(os.path.join(self.root_dir, self.annotations.iloc[index, 5]), self.annotations.iloc[index, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img*(1/255)
#         image = io.imread(img_path)
#         lab = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         lab_planes = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10, 10))
#         lab_planes[0] = clahe.apply(lab_planes[0])
#         lab = cv2.merge(lab_planes)
#         image = cv2.cvtColor(lab,cv2.COLOR_BGR2RGB)
        
        y_label = torch.tensor(int(self.annotations.iloc[index, 4]))
        
        try:  
            if self.transform is not None:
                img = self.transform(image = img)['image']
        except:
            print('error here in cutom Dataset transforming')
            print(img.shape)
            print(self.annotations.iloc[index, 0])

    
        return (img, y_label)

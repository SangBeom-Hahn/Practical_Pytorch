from base import BaseDataset
import pandas as pd
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import numpy as np

class IrisDataset(BaseDataset):
    """IrisDataset

    Args:
        BaseDataset (_type_): _description_
    """    
    
    def __init__(self, data_dir):
        self.data = pd.read_csv(data_dir + "/iris.csv")
        self.X = self.data.drop(["Id", "Species"], axis=1) # 넘파이 형식으로 반환
        self.y = self.data["Species"]
        self.transform = self._transforms()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X, y = None, None
        X = self.transform(X)
        y = torch.tensor(self.y[index], dtype=torch.long)
        
        return X, y
    
    def _transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
class ImageDataset(BaseDataset):
    def __init__(self, data_dir):
        # 이미지 데이터 초기화
        self.file_path = glob.glob(data_dir + "/*.jpg")
        self.X = np.array([plt.imread(self.file_path[idx]) for idx in range(len(self.file_path))])
        
        y = np.array([self.file_path[idx].split('\\')[-1].split('.')[0] for idx in range(len(self.file_path))])
        self.y = self._convert_to_numeric(y)
        
        self.transform = self._transforms()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):        
        X = self.X[index] / 255.
        X = self.transform(X)
        y = self.y[index]
        
        return X, torch.tensor(y, dtype=torch.long)
    
    def _convert_to_numeric(self, class_array):
        class_to_number = {class_value : idx for idx, class_value in enumerate(np.unique(class_array))}
        return np.vectorize(class_to_number.get)(class_array)
    
    def _transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
import os
from base import BaseDataset
import pandas as pd
import torch
from torchvision import transforms, datasets
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
        
class ImageFolderDataset(BaseDataset):
    # 이미지 RGB 값 평균으로 데이터 전처리
    def __init__(self, data_dir):
        self.transform = self._init_transforms()
        dataset = datasets.ImageFolder(data_dir, self._init_transforms())

        meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in dataset]
        stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in dataset]

        self.meanR = np.mean([m[0] for m in meanRGB])
        self.meanG = np.mean([m[1] for m in meanRGB])
        self.meanB = np.mean([m[2] for m in meanRGB])

        self.stdR = np.mean([s[0] for s in stdRGB])
        self.stdG = np.mean([s[1] for s in stdRGB])
        self.stdB = np.mean([s[2] for s in stdRGB])
        
        print("평균", self.meanR, self.meanG, self.meanB)
        print("표준편차", self.stdR, self.stdG, self.stdB)
        
        self.train_data = datasets.ImageFolder(os.path.join(data_dir), 
                                                self._main_transfomrs())
        
    def getDataset(self):
        return self.train_data
        
    # 픽셀별 정규화를 위한 증강
    def _init_transforms(self):
        return transforms.Compose([
            transforms.ToTensor()
        ])
        
    # 메인 증강
    def _main_transfomrs(self):
        return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 좌우반전 
        transforms.RandomVerticalFlip(),  # 상하반전 
        transforms.Resize((1024, 1024)),  # 알맞게 변경하세요 
        transforms.ToTensor(),  # 이 과정에서 [0, 255]의 범위를 갖는 값들을 [0.0, 1.0]으로 정규화, torch.FloatTensor로 변환
        transforms.Normalize([self.meanR, self.meanG, self.meanB], 
                                [self.stdR, self.stdG, self.stdB])  #  정규화(normalization)
    ])
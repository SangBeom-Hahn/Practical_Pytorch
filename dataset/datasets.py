from base import BaseDataset
import pandas as pd
import torch
from torchvision import transforms

class IrisDataset(BaseDataset):
    """IrisDataset

    Args:
        BaseDataset (_type_): _description_
    """    
    
    def __init__(self, data_dir):
        self.data = pd.read_csv(data_dir + "/iris.csv")
        self.X = self.data.drop(["Id", "Species"], axis=1) # 넘파이 형식으로 반환
        self.y = self.data["Species"]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        X, y = None, None
        X = self.transform(X)
        y = torch.tensor(self.y[index], dtype=torch.long)
        
        return X, y
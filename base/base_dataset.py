from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data_dir):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
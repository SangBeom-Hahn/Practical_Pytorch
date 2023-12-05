from base import BaseDataLoader
from torchvision import transforms

class IrisDataLoader(BaseDataLoader):
    """IrisDataLodaer

    Args:
        BaseDataLoader (_type_): _description_
    """
    
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = pass
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
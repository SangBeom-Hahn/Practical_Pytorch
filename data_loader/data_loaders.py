from base import BaseDataLoader
from torchvision import transforms
from dataset import Ir

class IrisDataLoader(BaseDataLoader):
    """IrisDataLodaer

    Args:
        BaseDataLoader (_type_): _description_
    """
    
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
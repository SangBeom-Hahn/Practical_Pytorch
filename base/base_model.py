import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """Base class for all models

    Args:
        nn (_type_): nn.Module
    """    
    
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError("모델을 구현해야 합니다.")
    
    def __str__(self):
        """Model prints with number of trainable parameters
        """        
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        params = len(list(model_parameters))
        return super().__str__() + "\nTrainable parameters: %d" % params
from pathlib import Path
import json
from collections import OrderedDict
import torch.nn as nn

def read_json(cfg_fname):
    fname = Path(cfg_fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook = OrderedDict)
    
def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
        
class OutputShapeHook(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        for name, layer in self.model.named_modules():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print("%s: %s" % (layer.__name__, str(output.shape)))
            )
            
    def forward(self, x):
        return self.model(x)
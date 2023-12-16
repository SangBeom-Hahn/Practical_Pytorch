from pathlib import Path
import json
from collections import OrderedDict
import torch.nn as nn
import torch

def read_json(cfg_fname):
    fname = Path(cfg_fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook = OrderedDict)
    
def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
        
def prepare_device(n_gpu_use):
    """GPU 사용이 가능하면 셋팅하는 메서드. DataParallel에 사용할 gpu 번호를 받는다.

    Args:
        n_gpu_use (_type_): 사용할 GPU 갯수

    Returns:
        device, list_ids (_type_): 사용할 장치(cpu, gpu), 사용할 GPU 번호
    """    
    n_gpu = torch.cuda.device_count()
    if(n_gpu_use > 0 and n_gpu == 0): # 사용한 GPU가 없으면
        print("현재 GPU를 사용할 수 없습니다.")
        n_gpu_use = 0
        
    if(n_gpu_use > n_gpu): # 사용할 갯수보다 사용가능한 gpu 개수가 적으면
        print("현재 할당가능한 GPU 갯수가 %d 보다 적습니다. \
            가용 가능한 개수만으로 학습을 돌립니다." % n_gpu_use)
        
        n_gpu_use = n_gpu
    # 위 조건에 따라서 최종 n_gpu_use를 결정했을 때 마지막으로 cpu를 쓸건지 gpu를 쓸 건지 결정
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu") # n_gpu_use가 0보다 크면 gpu를 쓴다.
    list_ids = list(range(n_gpu_use))
    return device, list_ids
        
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
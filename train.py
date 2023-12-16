import argparse
import collections
from config_parser import ConfigParser
import data_loader.data_loaders as module_data
from model.resnet34 import model as module_arch # 이거 model.py 꺼내야 겠다.
import model.loss as module_loss
import model.metric as module_metric

from utils import prepare_device
import torch
from trainer import Trainer

def main(config):
    logger = config.get_logger("train")
    data_loader = config.init_data_loader("data_loader", module_data)
    
    valid_data_loader = data_loader.split_validation()
    
    model = config.init_obj("arch", module_arch)
    logger.info(model) # info니깐 log 파일에 남음
    
    # config에 적힌 gpu 개수만큼 학습을 진행한다. 
    # 하지만 가용 가능한 GPU 개수가 적거나 없으면 CPU로 할 수도 있다.
    device, device_ids = prepare_device(config["n_gpu"]) 
    if(len(device_ids) > 1):
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, metric) for metric in config["metrics"]]
    
    # optm = optim.Adam(M.parameters(),lr=1e-3)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params) # optim에는 원래 파라미터가 들어가는데 학습 가능한 것만 넣기
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    trainer = Trainer(model, criterion, metrics, optimizer,
            config=config,
            device=device,
            data_loader=data_loader,
            valid_data_loader=valid_data_loader,
            lr_scheduler=lr_scheduler)
    
    trainer.train()
    

if(__name__ == "__main__"):
    args = argparse.ArgumentParser(description="Practical Pytorch")
    args.add_argument('-c', '--config', default=None, type=str,
                      help="config file path (default: None)")
    args.add_argument('-r', '--resume', default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    args.add_argument('-d', '--device', default=None, type=str,
                      help="indices of GPUs to enable (default: None)")
    
    CustomArgs = collections.namedtuple('CustomArgs', ['flags', "type", "target"])
    options = [
        CustomArgs(flags = ['--lr', '--learning_rate'], type = float, target = "optimizer;args;lr"),
        CustomArgs(flags = ['--bs', '--batch_size'], type = int, target = "data_loader;args;batch_size"),
    ]
    
    # config 파서 객체 만든 것을 main으로 전달
    config = ConfigParser.from_args(args, options)
    main(config)
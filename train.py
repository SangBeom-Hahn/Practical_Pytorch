import argparse
import collections
from config_parser import ConfigParser
import data_loader.data_loaders as module_data

def main(config):
    logger = config.get_logger("train")
    data_loader = config.init_data_loader("data_loader", module_data)
    
    # 여기까지 했고 테스트 ㄱㄱ
    

if(__name__ == "__main__"):
    args = argparse.ArgumentParser(description="Practical Pytorch")
    args.add_argument('-c', '--config', default=None, type=str,
                      help="config file path (default: None)")
    args.add_argument('-r', '--resume', default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    args.add_argument('-d', '--device', default=all, type=str,
                      help="indices of GPUs to enable (default: all)")
    
    CustomArgs = collections.namedtuple('CustomArgs', ['flags', "type", "target"])
    options = [
        CustomArgs(flags = ['--lr', '--learning_rate'], type = float, target = "optimizer;args;lr"),
        CustomArgs(flags = ['--bs', '--batch_size'], type = int, target = "data_loader;args;batch_size"),
    ]
    
    # config 파서 객체 만든 것을 main으로 전달
    config = ConfigParser.from_args(args, options)
    main(config)
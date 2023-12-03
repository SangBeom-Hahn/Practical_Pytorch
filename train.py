import argparse
import collections

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
        CustomArgs(['--lr', '--learning_rate'], type = float, target = "optimizer;args;lr"),
        CustomArgs(['--bs', '--batch_size'], type = int, target = "data_loader;args;batch_size"),
    ]
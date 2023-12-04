import os
import logging
from pathlib import Path
from utils import read_json, write_json
from datetime import datetime
from logger import setup_logging

class ConfigParser():
    def __init__(self, config, resume=None, modification=None, run_id=None) -> object:
        self.__config = _update_config(config, modification)
        self.resume = resume
        
        save_dir = Path(self.__config["trainer"]["save_dir"])
        exper_name = self.__config["name"]
        if(run_id is None):
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        # 모델, 로그 저장 경로 설정
        self.__save_dir = save_dir / "models" / exper_name / run_id
        self.__log_dir = save_dir / "log" / exper_name / run_id
        
        exist_ok = (run_id == '')
        # 게터로 속성 반환
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        
        write_json(self.config, self.save_dir / "config.json")
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
            
    @property
    def save_dir(self):
        return self.__save_dir
    
    @property
    def log_dir(self):
        return self.__log_dir
        
    @property
    def config(self):
        return self.__config
        
        
    
    '''
        options = [
            CustomArgs(flags = ['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
            CustomArgs(flags = ['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
        ]
    '''
    
    @classmethod
    def from_args(cls, args, options = ''): # None 대신 ""를 사용
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
            
        if (not isinstance(args, tuple)):
            args = args.parse_args()
        
        # GPU 설정
        if (args.device is not None):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            
        # config 파일 경로 설정
        if (args.resume is not None):
            resume = Path(args.resume)
            cfg_fname = resume.parent / "config.json"
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
            
        config = read_json(cfg_fname)
        if(args.config and resume):
            config.update(read_json(args.config))
            
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        
        return cls(config, resume, modification)
    
    # config 메서드로 불러온 OrderedDict를 접근함
    def __getitem__(self, name):
        return self.config[name]
    
def _get_opt_name(flags):
    for flg in flags:
        if(flg.startswith("--")):
            return flg.replace("--", '')
    return flags[0].replace("--", "")

def _update_config(config, modification):
    if(modification is None):
        return config
    
    # modification = {'optimizer;args;lr': None, 'data_loader;args;batch_size': None}
    for k, v in modification.items():
        if(v is not None):
            pass
    return config
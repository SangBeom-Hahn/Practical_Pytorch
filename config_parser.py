import os
import logging
from pathlib import Path
from utils import read_json, write_json
from datetime import datetime
from logger import setup_logging
import dataset.datasets as module_dataset

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
    
    def init_obj(self, name, module, *args, **kwargs):
        '''
            "data_loader": { 
                "type": "MnistDataLoader",
                "args":{
                    "data_dir": "data/",
                    "batch_size": 128,
                    "shuffle": true,
                    "validation_split": 0.1,
                    "num_workers": 2
                }
            }
        '''
        
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        
        # k가 module_args에 속하지 않는 것이 있다면 예외 발생
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        
        # getattr로 가져온 모듈에 callable로 호출하고 인자를 넣음
        return getattr(module, module_name)(*args, **module_args)
    
    def init_data_loader(self, name, module, *args, **kwargs):
        module_name = self[name]["type"]
        # 데이터 셋 추출
        dataset = self.init_obj("dataset", module_dataset)
        
        # 테스트를 위해 잠깐 넣음
        image, label = next(iter(dataset))
        
        # 데이터 로더 생성자에 데이터 셋을 가장 먼저 넣기
        module_args = {"dataset" : dataset} 
        module_args.update(self[name]['args'])
        
        return getattr(module, module_name)(*args, **module_args)
        
    
    # config 메서드로 불러온 OrderedDict를 접근함
    def __getitem__(self, name):
        return self.config[name]
    
    def get_logger(self, name, verbosity = 2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
        
    
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
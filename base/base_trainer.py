import torch
from abc import abstractclassmethod
from numpy import inf

class BaseTrainer():
    """
    Base Trainer
    """
    
    def __init__(self, model, criterion, metric_ftns, optimizer, config) -> None:
        self.config = config
        self.logger = config.get_logger('trainer', config["trainer"]["verbosity"])
        self.model = model
        
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        
        cfg_trainer = config["trainer"] # config.json에 있는 파라미터 선언
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off") # 기본 값 off
        
        if(self.monitor == "off"):
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split() # "min val_loss"를 split
            assert self.mnt_mode in ["min", "max"]
            
            # best 모델을 저장할 건데 mode가 최소값이면 best 값을 무한대로
            self.mnt_best = inf if self.mnt_mode == "min" else -inf 
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if(self.early_stop <= 0):
                self.early_stop = inf
                
        self.start_epoch = 1
        # dir이 config parser에 있음
        self.checkpoint_dir = config.save_dir
        
        # resume config parser에 있음
        if(config.resume is not None):
            self._resume_checkpoint(config.resume) # 현재는 resume None
            
    @abstractclassmethod
    def _train_epoch(self, epoch):
        """Training logic

        Args:
            epoch (_type_): 에폭 수
        """        
        raise NotImplementedError("훈련 로직을 구현해야 합니다.")
    
    def train(self):
        """Train logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch, self.epochs)
            # 하는 중
        print ("Done")
            
    def _resume_checkpoint(self, resume_path):
        """
        이슈로 훈련이 중단된 경우 저장된 모델을 가지고 모델 학습을 재개한다.

        Args:
            resume_path (_type_): 체크포인트 저장 경로
        """        
        
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ..." % resume_path)
        checkpoint = torch.load(resume_path)
        
        '''
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint("epoch")
        '''
        
        # 위에 가중치, 모델 로드하는 거 readme에 기록해뒀는데 checkpoint를 사전처럼 접근할 수 있나보다
        # resume을 하면 이전 에폭, 가중치에서 이어서 학습한다.
        self.start_epoch = checkpoint["epoch"] + 1 
        self.mnt_best = checkpoint["monitor_best"]
        
        if(checkpoint["config"]["arch"] != self.config["arch"]):
            self.logger.warning("checkpoint 가중치 파일의 기존 모델과 가중치를 장착할 모델 구조가 다릅니다.")
        self.model.load_state_dict(checkpoint["state_dict"])
        
        if(checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]):
            self.logger.warning("checkpoint 가중치 파일의 기존 옵티마이저와 현재 옵티마이저가 다릅니다.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
        self.logger.info("checkpoint loaded. Resume training from epoch {}" % self.start_epoch)
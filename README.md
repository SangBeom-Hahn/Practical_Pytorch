# Practical_Pytorch
실전 Pytorch 구축 공간

## Introduction

본 프로젝트는 pytorch 프레임워크를 활용하여 데이터셋, 데이터 로더, 트레이너, 모델을 적재적소로 커스터마이징할 수 있도록 합니다. AI 연구자 및 엔지니어는 원하는 모듈을 구현하고 config를 수정하면 클라이언트 코드 변경 없이 기능을 추가할 수 있는 OCP 원칙을 달성할 수 있습니다.

## Project Structure

```
Practical_Pytorch
├── base
├── data_loader
├── dataset
├── logger
├── model
├── trainer
├── utils
├── config.json
├── config_parser.py
├── inference.py
├── kfold_train.py
├── train.py
└── requirements.txt
```

- base : 데이터셋, 데이터 로더, 트레이너, 모델 상위 super 클래스
- data_loader : 사용할 커스텀 데이터 로더
- dataset : 커스텀 데이터 셋
- model : 커스텀 모델
- trainer : 커스텀 트레이너
- logger : 로깅
- utils : json 파싱, GPU 유틸리티
- config.json : train 셋팅 cfg
- config_parser.py : cfg 파싱 컴포넌트
- inference.py : 추론 코드
- train.py : 학습 코드
- kfold_train : kfold 학습 코드

## Features
- config.json으로 편리하게 매개변수 변경
- base 추상 클래스 사용
- 명확한 폴더 구조로 협업 시 충돌 문제 제거

## Usage
- config.json 수정
```
{
    "name": "Mask_ResNet",
    "n_gpu": 1,

    "arch": {
        "type": "ResNet",
        "args": {
            "num_blocks": [3, 4, 6, 3]
        }
    },
    "data_loader": { 
        "type": "IrisDataLoader",
        "args":{
            "batch_size": 3,
            "shuffle": false,
            "validation_split": 2,
            "num_workers": 2
        }
    },
    "dataset": {
        "type": "ImageDataset",
        "args":{
            "data_dir" : "data/"
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "nll_loss",
    "metrics": [
        "accuracy", "top_k_acc"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 1,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        
        "monitor": "min val_loss",
        "early_stop": 10,
        "tensorboard": true
    }
}
```

- 모델 학습 : ```python train.py -c config.json```
- 인퍼런스 : ```python train.py --resume path/to/checkpoint```
  - 모델 학습 결과 results dir에 pth 가중치 파일이 저장됩니다.

## Customization
### Custom ConfigParser
```python
data_loader = config.init_data_loader("data_loader", module_data)
model = config.init_obj("arch", module_arch)

epochs = config["epochs"]
```
config.json을 파싱하고 편리하게 사용할 수 있는 모듈입니다. 내부 메서드로 모델 학습 컴포넌트를 추출할 수 있고 _getItem_을 재정의하여 키워드로 접근할 수 있습니다.

### Data Loader
1. ```BaseDataLoader``` 상속
BaseDataLoader.split_validation()으로 훈련, 검증 셋을 나눠서 데이터 로더를 사용할 수 있습니다.

2. 사용법
```python
data_loader = config.init_data_loader("data_loader", module_data)

for batch_idx, (x_batch, y_batch) in data_loader:
    pass
```

3. 커스터 마이징
```data_loader/data_loaders```에 원하는 데이터 로더를 구현하고 configparser로 주입받아 사용할 수 있습니다.

### Dataset
1. ```BaseDataset``` 상속
2. 사용법
```python
def init_data_loader(self, name, module, *args, **kwargs):
        module_name = self[name]["type"]
        # 데이터 셋 추출
        dataset = self.init_obj("dataset", module_dataset)
```
config parser의 데이터 로더 생성 메서드 내부에서 데이터 셋을 생성하고 인자로 주어 원하는 데이터 셋, 로더 쌍을 사용하도록 하였으며, config.json으로 선택할 수 있습니다.

### Trainer
1. ```BaseTrainer``` 상속
2. 사용법
```python
def train(self):
        """Train logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch, self.epochs)

trainer.train()
```
BaseTrainer의 train 메서드가 커스텀 Trainer의 _train_epoch를 호출합니다. main 메서드에서 커스텀 트레이너를 임포트하면 하나의 훈련 시나리오에 종속되지 않게 모델 학습을 할 수 있습니다.

### Model
1. 사용법
```model/```에 커스텀 모델을 만들고

```
"arch": {
    "type": "ResNet",
    "args": {
        "num_blocks": [3, 4, 6, 3]
    }
}
```
config.json으로 주입받아서 사용할 수 있습니다.

### Logging
1. Setup Logger
```logger/logger```의 setup_logging 메서드에서 python logging을 셋팅합니다.
```logger_config.json```에 setting 데이터가있습니다.

```python
logging.config.dictConfig(config)
```
dictConfig로 세세한 로거 셋팅을 할 수 있습니다.

2. 사용법
```python
logger = config.get_logger("train")
logger.info(model)
```

로깅을 원하는 모듈에서 주입받아서 사용할 수 있습니다.

### Wandb
1. Setup Wandb
```python
wandb.init(project="practical-pytorch", config={})
```
train.py에서 wandb를 셋팅합니다.
   
2. 사용법
```
wandb.log({
    "train_accr" : train_accr,
    "val_accr" : val_accr
}, step = epoch)
```
trainer.py에 학습 코드에서 학습 모니터링을 할 수 있습니다.

### Additional Utils
- Early Stopping
- Lr Scheduler
- K-fold Train

from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
import argparse
import data_loader.data_loaders as module_data
import collections
import dataset.datasets as module_dataset
from model import model as module_arch
import model.loss as module_loss
from config_parser import ConfigParser
import torch
import os
import numpy as np

batch_size = 64
num_workers = 4
num_classes = 18

num_epochs = 1  # 학습할 epoch의 수
lr = 1e-4
lr_decay_step = 10
criterion_name = 'cross_entropy' # loss의 이름

train_log_interval = 20  # logging할 iteration의 주기
name = "02_model_results"  # 결과를 저장하는 폴더의 이름

# -- settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def getConfig():
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
    return ConfigParser.from_args(args, options)

def getDataloader(config):
    data_loader = config.init_data_loader("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()
    
    return data_loader, valid_data_loader

def getModel(config):
    return config.init_obj("arch", module_arch)


def kFoldTrain(config):
    os.makedirs(os.path.join(os.getcwd(), 'results', name), exist_ok=True)

    # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
    n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits)

    counter = 0
    patience = 10
    accumulation_steps = 2
    best_val_acc = 0
    best_val_loss = np.inf
    dataset = config.init_obj("dataset", module_dataset)
    labels = dataset.y

    # Stratified KFold를 사용해 Train, Valid fold의 Index를 생성합니다.
    # labels 변수에 담긴 클래스를 기준으로 Stratify를 진행합니다.
    # 매 이터레이션 총 k개
    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_list, labels)):

        # 생성한 Train, Valid Index를 getDataloader 함수에 전달해 train/valid DataLoader를 생성합니다.
        # 생성한 train, valid DataLoader로 이전과 같이 모델 학습을 진행합니다.
        train_loader, val_loader = getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers)

        # -- model
        model = config.init_obj("arch", module_arch)

        # -- loss & metric
        criterion = getattr(module_loss, config["loss"])
        train_params = [{'params': getattr(model, 'features').parameters(), 'lr': lr / 10, 'weight_decay':5e-4},
                        {'params': getattr(model, 'classifier').parameters(), 'lr': lr, 'weight_decay':5e-4}]
        optimizer = Adam(train_params)
        
        for epoch in range(num_epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()

                # -- Gradient Accumulation
                if (idx+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % train_log_interval == 0:
                    train_loss = loss_value / train_log_interval
                    train_acc = matches / batch_size / train_log_interval
                    print(
                        f"Epoch[{epoch}/{num_epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
                    )

                    loss_value = 0
                    matches = 0

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(valid_idx)

                # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_acc > best_val_acc:
                    print("New best model for val accuracy! saving the model..")
                    torch.save(model.state_dict(), f"results/{name}/{epoch:03}_accuracy_{val_acc:4.2%}.ckpt")
                    best_val_acc = val_acc
                    counter = 0
                else:
                    counter += 1
                # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
                if counter > patience:
                    print("Early Stopping...")
                    break

                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
            
if(__name__ == "__main__"):
    config = getConfig()
    kFoldTrain(config)
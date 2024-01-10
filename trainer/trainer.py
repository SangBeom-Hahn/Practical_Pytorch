from base import BaseTrainer
import torch
import wandb
import numpy as np

class Trainer(BaseTrainer):
    """Trainer class

    Args:
        BaseTrainer (_type_): Parent base Trainer
    """    
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None) -> None:
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.data_loader = data_loader
        self.device = device
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
    
    def _train_epoch(self, epoch, EPOCHS):
        """Training logic for an epoch

        Args:
            epoch (_type_): training epoch
        """
        self.model.train() # 학습 모드
        loss_value = 0
        matches = 0
        
        for idx, train_batch in enumerate(self.data_loader):
            batch_in, batch_out = train_batch
            batch_in = batch_in.float().to(self.device)
            self.optimizer.zero_grad()
            
            y_pred = self.model.forward(batch_in)
            y_pred = torch.argmax(y_pred, dim=-1)

            loss_out = self.criterion(y_pred.float(), batch_out.long().to(self.device))
            
            loss_out.backward()
            self.optimizer.step()
            
            loss_value += loss_out.item()
            matches += (y_pred == batch_out).sum().item()
            if (idx + 1) % 20 == 0:
                train_loss = loss_value / 20
                train_acc = matches / 32 / 20 # args.batch_size / args.log_interval
                
                print(
                    f"Epoch[{epoch}/{EPOCHS}]({idx + 1}/{len(self.data_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
                )
                loss_value = 0
                matches = 0
                
        if(self.lr_scheduler is not None):
            self.lr_scheduler.step()
           
        self._func_eval(self.model, self.valid_data_loader, self.device)

    def _func_eval(self, model, data_loader, device):
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            
            for batch_in, batch_out in data_loader:
                batch_in = batch_in.float().to(device)
                y_target = batch_out.long().to(device)
                model_pred = model(batch_in)
                
                # 모델 예측 값이 10개의 클래스라면 가장 높은 확률의 인덱스를 알려줌
                _, y_pred = torch.max(model_pred.data, 1)
                
                loss_item = self.criterion(model_pred, y_target).item()
                acc_item = (y_target == y_pred).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
            val_loss = np.sum(val_loss_items) / len(data_loader)
            val_acc = np.sum(val_acc_items) / len(data_loader) # set
            iteration_change_loss += 1
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{self.checkpoint_dir}/best.pth")
                best_val_acc = val_acc
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                iteration_change_loss = 0
            
            torch.save(model.module.state_dict(), f"{self.checkpoint_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            
            model.train()
    
    # def _show_model_pred(): # 물론 test.py가 있겠지만 이거로 단순하게 모델 예측값 저자애보고 싶다.
    #     n_sample = 25
    #     sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
    #     test_x = mnist_test.data[sample_indices]
    #     test_y = mnist_test.targets[sample_indices]
    #     with torch.no_grad():
    #         y_pred = M.forward(test_x.view(-1, 28*28).type(torch.float).to(device)/255.)
    #     y_pred = y_pred.argmax(axis=1)
    #     plt.figure(figsize=(10,10))
    #     for idx in range(n_sample):
    #         plt.subplot(5, 5, idx+1)
    #         plt.imshow(test_x[idx], cmap='gray')
    #         plt.axis('off')
    #         plt.title("Pred:%d, Label:%d"%(y_pred[idx],test_y[idx]))
    #     plt.show()
    #     print ("Done")
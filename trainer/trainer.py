from base import BaseTrainer
import torch

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
        print_every = 1
        loss_val_sum = 0
        for batch_in, batch_out in self.data_loader:
            batch_in = batch_in.float().to(self.device)
            y_pred = self.model.forward(batch_in)
            loss_out = self.criterion(y_pred, batch_out.long().to(self.device))
            
            self.optimizer.zero_grad()
            loss_out.backward()
            self.optimizer.step()
            loss_val_sum += loss_out
        loss_val_avg = loss_val_sum / len(self.data_loader)
        
        # Print
        if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
            train_accr = self._func_eval(self.model, self.data_loader, self.device)
            test_accr = self._func_eval(self.model, self.valid_data_loader, self.device)
            
            print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
                (epoch, loss_val_avg, train_accr, test_accr))
            
        if(self.lr_scheduler is not None):
            self.lr_scheduler.step()
            
    def _func_eval(self, model, data_loader, device):
        with torch.no_grad():
            model.eval()
            n_total, n_correct = 0, 0
            for batch_in, batch_out in data_loader:
                batch_in = batch_in.float().to(device)
                y_target = batch_out.long().to(device)
                model_pred = model(batch_in)
                
                # 모델 예측 값이 10개의 클래스라면 가장 높은 확률의 인덱스를 알려줌
                _, y_pred = torch.max(model_pred.data, 1)
                
                n_correct += (
                    y_target == y_pred
                ).sum().item()
                n_total += batch_in.size(0)
                
            val_accr = n_correct / n_total
            model.train()
        return val_accr
    
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
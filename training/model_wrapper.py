import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
from torch.optim.lr_scheduler import ExponentialLR

class ModelWrapper(nn.Module):
    '''
    Класс для работы с моделью
    '''
    def __init__(self, model: object):
        super(ModelWrapper, self).__init__()
        self.model = model

        
    def forward(self, input_data):
        output_data = self.model(input_data)
        return output_data
    
    
    def fit(self, criterion: object, metric: object, optimizer: object, scheduler: object,
                  train_data_loader: DataLoader, valid_data_loader: DataLoader=None, 
                  epochs: int=1, verbose: int=5):        
        '''Метод для обучения модели
        Входные параметры:
        criterion: object - объект для вычисления loss
        metric: object - объект для вычисления метрики качества
        optimizer: object - оптимизатор
        scheduler: object - объект для изменения lr в ходе обучения
        train_data_loader: DataLoader - загрузчик данных для обучения
        valid_data_loader: DataLoader - загрузчик данных для валидации
        epochs: int - количество эпох обучения
        verbose: int - вывод информации через каждые verbose итераций
        Возвращаемые значения:
        result: dict - словарь со значениями loss при тренировке, валидации и метрики при валидации 
        для каждой эпохи'''
        self.optimizer = optimizer
        epoch_train_losses = []
        epoch_valid_losses = []
        epoch_valid_metrics = []
        result = {}
        
        for epoch in range(epochs):
            self.model.train()
            time1 = time.time()
            running_loss = 0.0
            train_losses = []
            
            for batch_idx, data in enumerate(train_data_loader):

                inputs, target = data[0], data[1]
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs[:,0], target.to(torch.float32))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_losses.append(loss.item())
                if (batch_idx+1) % verbose == 0:
                    print(f'Train Epoch: {epoch+1}, Loss: {(running_loss/verbose):.6f}, ', end="")
                    print(f'Learning rate: {scheduler.get_last_lr()[0]}')
                    time2 = time.time()
                    print(f'Spended time for {verbose} batches (total: {batch_idx})({int((verbose*data[0].shape[0]))} images', end="") 
                    print(f': {(time2-time1):.6f} sec')
                    
                    time1 = time.time()
                    running_loss = 0.0
                if (batch_idx+1) % 3000 == 0:
                    train_loss = np.mean(train_losses)
                    if valid_data_loader != None:
                        valid_result = self.valid(criterion, metric, valid_data_loader)
                        valid_loss = valid_result['valid_loss']
                        valid_metric = valid_result['valid_metric']

                        print('='*80)
                        print(f'Epoch {epoch+1}, train loss: {(train_loss):.6f}, valid loss: {(valid_loss):.6f}, ', end="")
                        print(f'valid metric: {(valid_metric):.6f}')
                        print('='*80)
                    else:
                        print('='*80)
                        print(f'Epoch {epoch+1}, train loss: {(train_loss):.6f}')
                        print('='*80)
                        valid_loss = None
                        valid_metric = None

            train_loss = np.mean(train_losses)
            scheduler.step()
            
            if valid_data_loader != None:
                valid_result = self.valid(criterion, metric, valid_data_loader)
                valid_loss = valid_result['valid_loss']
                valid_metric = valid_result['valid_metric']

                print('='*80)
                print(f'Epoch {epoch+1}, train loss: {(train_loss):.6f}, valid loss: {(valid_loss):.6f}, ', end="")
                print(f'valid metric: {(valid_metric):.6f}')
                print('='*80)
            else:
                print('='*80)
                print(f'Epoch {epoch+1}, train loss: {(train_loss):.6f}')
                print('='*80)
                valid_loss = None
                valid_metric = None
            epoch_train_losses.append(train_loss)
            epoch_valid_losses.append(valid_loss)
            epoch_valid_metrics.append(valid_metric)
        
        result['epoch_train_losses'] = epoch_train_losses
        result['epoch_valid_losses'] = epoch_valid_losses
        result['epoch_valid_metrics'] = epoch_valid_metrics
        
        return result
    
    
    def valid(self, criterion: object, metric: object, valid_data_loader: DataLoader):
        '''Метод для валидации модели
        Входные параметры:
        criterion: object - объект для вычисления loss
        metric: object - объект для вычисления метрики качества
        valid_data_loader: DataLoader - загрузчик данных для валидации
        Возвращаемые значения:
        result: dict - словарь со значениями loss и метрики при валидации'''
        self.model.eval()
        valid_metrics = []
        valid_losses = []
        result = {}
        y_true = []
        y_proba = []
        for batch_idx, data in enumerate(valid_data_loader):
            inputs = data[0]
            target = data[1]
            outputs = self.model(inputs)        
            loss = criterion(outputs[:, 0], target.to(torch.float32))
            valid_losses.append(loss.item())
            y_true.append(np.array(target.detach().numpy()))
            y_proba.append(np.array(outputs[:,0].detach().numpy()))
        y_true = np.concatenate(np.array(y_true), axis=0)
        y_proba = np.concatenate(np.array(y_proba), axis=0)
        valid_loss    = np.mean(valid_losses)
        valid_metric  = metric(y_true, y_proba)
        result['valid_loss'] = valid_loss
        result['valid_metric'] = valid_metric
        self.model.train()
        return result
    
    
    def save(self, path_to_save: str = './data/checkpoint.pth'):
        torch.save(self.model.state_dict(), path_to_save)


    def load(self, path_to_model: str = './data/checkpoint.pth'):
        self.model.load_state_dict(torch.load(path_to_model))
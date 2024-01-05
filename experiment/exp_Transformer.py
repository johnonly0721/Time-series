import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler

from experiment.exp_basic import ExperimentBasic
from models import Informer, Transformer
from utils.data.data_factory import data_provider
from utils.metrics import metrics
from utils.tools import (EarlyStopping, Timer,
                         adjust_learning_rate, test_params_flop, visual)

warnings.filterwarnings('ignore')


class Exp_Transformer(ExperimentBasic):
    def __init__(self, config):
        super().__init__(config)

    def _build_model(self):
        models = {
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = models.get(self.config.model, Transformer).Model(
            self.config).float()

        if self.config.use_gpu and self.config.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.config.devices)
        self.model = model.to(self.device)

    def _load_data(self, data_type):
        data_set, data_loader = data_provider(self.config, data_type)
        return data_set, data_loader

    def _get_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def _get_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        _, train_loader = self._load_data('train')
        val_data, val_loader = self._load_data('val')
        test_data, test_loader = self._load_data('test')

        train_steps_per_epoch = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.config.patience, verbose=True)

        optimizer = self._get_optimizer()
        criterion = self._get_criterion()

        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        model_save_path = os.path.join(self.config.checkpoints, setting)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, steps_per_epoch=train_steps_per_epoch,
                                            pct_start=self.config.pct_start, max_lr=self.config.lr,
                                            epochs=self.config.epochs)

        wandb.watch(self.model, criterion, log='all', log_freq=10)

        epoch_timer = Timer()
        epoch_timer.start()
        for epoch in range(self.config.epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            iter_timer = Timer()
            iter_timer.start()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        _, _, loss = self.calculate_one_batch(
                            batch_x, batch_x_mark, batch_y, batch_y_mark, criterion, mode='train')
                        train_loss.append(loss.item())

                else:
                    _, _, loss = self.calculate_one_batch(
                        batch_x, batch_x_mark, batch_y, batch_y_mark, criterion, mode='train')
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = iter_timer.stop() / iter_count
                    left_time = speed * \
                        ((self.config.epochs - epoch)
                         * train_steps_per_epoch - i)
                    left_hours = int(left_time // 3600)
                    left_minutes = int(left_time - 3600 * left_hours) // 60
                    left_seconds = int(left_time - 3600 *
                                       left_hours - 60 * left_minutes)
                    print("\tspeed: {:.4f}s/iter; left time: {}h {}min {}s".format(
                        speed, left_hours, left_minutes, left_seconds))
                    iter_count = 0
                    iter_timer.start()

                if self.config.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if self.config.lradj == 'TST':
                    adjust_learning_rate(
                        optimizer, scheduler, epoch + 1, self.config, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {epoch_timer.stop()}s")
            train_loss = np.average(train_loss)
            val_loss = self.validate(val_data, val_loader, criterion)
            test_loss = self.validate(test_data, test_loader, criterion)

            wandb.log({'train_loss': train_loss,
                      'val_loss': val_loss, 'test_loss': test_loss})

            epoch_timer.start()

            print("Epoch: {}, Steps: {} | train_loss: {:.7f}, val_loss: {:.7f}, test_loss: {:.7f}".format(
                epoch + 1, train_steps_per_epoch, train_loss, val_loss, test_loss))
            early_stopping(val_loss, self.model, model_save_path)
            if early_stopping.early_stop:
                print("提前停止训练")
                break

            if self.config.lradj != 'TST':
                adjust_learning_rate(optimizer, scheduler,
                                     epoch + 1, self.config)
            else:
                print(f'更新学习率为： {scheduler.get_last_lr()[0]}')

        best_model_path = model_save_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def validate(self, val_data, val_loader, criterion):
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    pred, true, loss = self.calculate_one_batch(
                        batch_x, batch_x_mark, batch_y, batch_y_mark, criterion, mode='val')
            else:
                pred, true, loss = self.calculate_one_batch(
                    batch_x, batch_x_mark, batch_y, batch_y_mark, criterion, mode='val')
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._load_data('test')

        if test:
            print('加载模型...')
            self.model.load_state_dict(torch.load(os.path.join(
                './checkpoints/' + setting + 'ckeckpoint.pth')))

        preds = []
        trues = []
        input_x = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        output, batch_y, _ = self.calculate_one_batch(
                            batch_x, batch_x_mark, batch_y, batch_y_mark, criterion=None, mode='test')
                else:
                    output, batch_y, _ = self.calculate_one_batch(
                        batch_x, batch_x_mark, batch_y, batch_y_mark, criterion=None, mode='test')

                output = output.numpy()
                batch_y = batch_y.numpy()

                preds.append(output)
                trues.append(batch_y)
                input_x.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate(
                        (input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate(
                        (input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        if self.config.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        input_x = np.array(input_x)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        input_x = input_x.reshape(-1, input_x.shape[-2], input_x.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        metrics_dict = metrics(preds, trues)
        print(
            f'mse: {metrics_dict["MSE"]}, mae: {metrics_dict["MAE"]}, rse: {metrics_dict["RSE"]}')
        with open('result.txt', 'a') as f:
            f.write(setting + ' \n')
            f.write(
                f'mse: {metrics_dict["MSE"]}, mae: {metrics_dict["MAE"]}, rse: {metrics_dict["RSE"]}\n')
            f.write('\n')
            f.close()
        np.save(folder_path + 'pred.npy', preds)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data('pred')

        if load:
            path = os.path.join(self.config.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                # encoder - decoder
                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.infer(
                            batch_x, batch_y, batch_x_mark, batch_y_mark)
                else:
                    outputs = self.infer(
                        batch_x, batch_y, batch_x_mark, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

    def infer(self, x, y, x_mark, y_mark):
        x = x.float().to(self.device)
        y = y.float().to(self.device)
        x_mark = x_mark.float().to(self.device)
        y_mark = y_mark.float().to(self.device)

        dec_input = torch.zeros_like(y[:, -self.config.pred_len:, :]).float()
        dec_input = torch.cat(
            [y[:, :self.config.label_len, :], dec_input], dim=1).float().to(self.device)

        if self.config.output_attention:
            outputs = self.model(
                x, x_mark, dec_input, y_mark)[0]
        else:
            outputs = self.model(
                x, x_mark, dec_input, y_mark)

        f_dim = -1 if self.config.features == 'MS' else 0
        outputs = outputs[:, -self.config.pred_len:, f_dim:]

        return outputs

    def calculate_one_batch(self, batch_x, batch_x_mark, batch_y, batch_y_mark, criterion, mode='train'):
        if mode == 'train':
            self.model.train()
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            dec_input = torch.zeros_like(
                batch_y[:, -self.config.pred_len:, :]).float()
            dec_input = torch.cat(
                [batch_y[:, :self.config.label_len, :], dec_input], dim=1).float().to(self.device)

            if self.config.output_attention:
                outputs = self.model(batch_x, batch_x_mark,
                                     dec_input, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark,
                                     dec_input, batch_y_mark)

            f_dim = -1 if self.config.features == 'MS' else 0
            outputs = outputs[:, -self.config.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.config.pred_len:,
                              f_dim:].float().to(self.device)

            loss = criterion(outputs, batch_y)

            return outputs, batch_y, loss
        else:
            self.model.eval()
            with torch.no_grad():
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_input = torch.zeros_like(
                    batch_y[:, -self.config.pred_len:, :]).float()
                dec_input = torch.cat(
                    [batch_y[:, :self.config.label_len, :], dec_input], dim=1).float().to(self.device)

                if self.config.output_attention:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_input, batch_y_mark)[0]
                else:
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_input, batch_y_mark)

                f_dim = -1 if self.config.features == 'MS' else 0
                outputs = outputs[:, -self.config.pred_len:,
                                  f_dim:].detach().cpu()
                batch_y = batch_y[:, -self.config.pred_len:,
                                  f_dim:].float().detach().cpu()

                if criterion is not None:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = None

                return outputs, batch_y, loss

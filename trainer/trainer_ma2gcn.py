from utils import masked_mae, masked_mape, masked_rmse
import numpy as np
import torch.nn as nn
import torch


class trainer:
    def __init__(self, model, optimizer, scaler, theta, io_range, io_adj):
        self.model = model
        self.criterion = masked_mae
        self.optimizer = optimizer
        self.scaler = scaler
        self.theta = theta
        self.io_range = io_range
        self.io_adj = io_adj
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def train(self, x, y_real, cur_io_adj):
        self.optimizer.zero_grad()

        output, cur_io_adj = self.model((x, cur_io_adj))
        y_pred = self.scaler.inverse_transform(output)
        loss = self.theta * self.criterion(y_pred[..., 0], y_real[..., 0], 0.0) + \
               (1 - self.theta) * self.criterion(y_pred[..., 1], y_real[..., 1], 0.0)

        loss.backward()
        self.optimizer.step()

        mape = masked_mape(y_pred, y_real, 0.0).item()
        rmse = masked_rmse(y_pred, y_real, 0.0).item()
        return loss.item(), mape, rmse

    def train_one_epoch(self, train_loader):
        self.model.train()
        loss_inner_epoch = []
        mape_inner_epoch = []
        rmse_inner_epoch = []
        for batch_idx, (x, y_real) in enumerate(train_loader):
            batch_num, _, _, _ = x.shape
            batch_start = batch_idx * batch_num
            batch_end = (batch_idx + 1) * batch_num
            io_start = np.digitize(batch_start, self.io_range) - 1
            io_end = np.digitize(batch_end, self.io_range) - 1
            cur_io_adj = (self.io_adj[io_start] + self.io_adj[io_end]) / 2
            loss, mape, rmse = self.train(x, y_real, cur_io_adj)
            loss_inner_epoch.append(loss)
            mape_inner_epoch.append(mape)
            rmse_inner_epoch.append(rmse)

        return np.array(loss_inner_epoch).mean(), np.array(mape_inner_epoch).mean(), \
            np.array(rmse_inner_epoch).mean()

    def eval(self, x, y_real, cur_io_adj):
        output, cur_io_adj = self.model((x, cur_io_adj))
        y_pred = self.scaler.inverse_transform(output)

        loss = self.theta * self.criterion(y_pred[..., 0], y_real[..., 0], 0.0) + \
               (1 - self.theta) * self.criterion(y_pred[..., 1], y_real[..., 1], 0.0)

        mape = masked_mape(y_pred, y_real, 0.0).item()
        rmse = masked_rmse(y_pred, y_real, 0.0).item()
        return loss.item(), mape, rmse

    def eval_one_epoch(self, val_loader, cur_io_adj):
        self.model.eval()
        loss_inner_epoch = []
        mape_inner_epoch = []
        rmse_inner_epoch = []
        for batch_idx, (x, y_real) in enumerate(val_loader):
            loss, mape, rmse = self.eval(x, y_real, cur_io_adj)
            loss_inner_epoch.append(loss)
            mape_inner_epoch.append(mape)
            rmse_inner_epoch.append(rmse)

        return np.array(loss_inner_epoch).mean(), np.array(mape_inner_epoch).mean(), \
            np.array(rmse_inner_epoch).mean()

    def test(self, test_loader, cur_io_adj):
        loss_inner_epoch = []
        mape_inner_epoch = []
        rmse_inner_epoch = []
        with torch.no_grad():
            for batch_idx, (x, y_real) in enumerate(test_loader):
                loss, mape, rmse = self.eval(x, y_real, cur_io_adj)
                loss_inner_epoch.append(loss)
                mape_inner_epoch.append(mape)
                rmse_inner_epoch.append(rmse)

        return np.array(loss_inner_epoch).mean(), np.array(mape_inner_epoch).mean(), \
            np.array(rmse_inner_epoch).mean()

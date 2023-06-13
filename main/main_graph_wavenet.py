import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from models.graph_wavenet import gwnet
from utils import ProcessData, StandardScalar
from torch.utils.data import DataLoader, TensorDataset
from utils import masked_mae
from trainer.trainer_graph_wavenet import trainer


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    # ------------------------ Data loading ------------------------
    data_config = config['dataset']
    optim_config = config['optimize']
    trainer_config = config['trainer']
    feature_matrix = np.load(data_config['feature_matrix_data_path'])
    samples = feature_matrix.shape[0]
    # shape -> (number of samples, sequence length, number of vertexes, features)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = ProcessData(feature_matrix,
                                                                                         data_config['input_seq'],
                                                                                         data_config['output_seq'],
                                                                                         data_config['split_rate'])
    # data scalar
    device = data_config['device']
    scalar = StandardScalar(train_data, device, axis=(0, 1, 2))
    data_x_sets = [train_data, val_data, test_data]
    data_y_sets = [train_labels, val_labels, test_labels]
    data_x_normed_sets = []
    data_y_real_sets = []
    for data in data_x_sets:
        data_normed = scalar.transform(data)
        data_normed = np.einsum('btnf->bfnt', data_normed)
        data_normed = torch.Tensor(data_normed).to(device)
        data_x_normed_sets.append(data_normed)
    train_x, val_x, test_x = data_x_normed_sets
    for data in data_y_sets:
        data_real = torch.Tensor(data).to(device)
        data_y_real_sets.append(data_real)
    train_real_y, val_real_y, test_real_y = data_y_real_sets
    data_final_list = [train_x, train_real_y, val_x, val_real_y, test_x, test_real_y]

    # get dataloader
    data_loaders = []
    for i in range(0, len(data_final_list), 2):
        dataset = TensorDataset(data_final_list[i], data_final_list[i + 1])
        data_loader = DataLoader(dataset, batch_size=data_config['batch_size'])
        data_loaders.append(data_loader)

    # get adjacency matrix
    origin_adj = pd.read_csv(data_config['adj_data_path'], header=None)
    origin_adj = torch.Tensor(origin_adj.values).to(device)

    # ------------------------ Model setting ------------------------
    supports = [origin_adj]
    if trainer_config['randomadj']:
        adjinit = None
    else:
        adjinit = supports[0]
    if trainer_config['aptonly']:
        supports = None

    model = gwnet(device, data_config['num_of_vertices'], trainer_config['dropout'], supports,
                  trainer_config['gcn_bool'], trainer_config['addaptadj'], adjinit,
                  trainer_config['in_dim'], trainer_config['out_dim'], trainer_config['nhid'],
                  trainer_config['nhid'], trainer_config['nhid'] * 8, trainer_config['nhid'] * 16)
    model.to(device)

    if optim_config['name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=optim_config['learning_rate'],
                               weight_decay=optim_config['weight_decay'])

    elif optim_config['name'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=optim_config['learning_rate'],
                                weight_decay=optim_config['weight_decay'])

    elif optim_config['name'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=optim_config['learning_rate'],
                                  weight_decay=optim_config['weight_decay'])
    else:
        raise NotImplementedError(f'ERROR: The optimizer {optim_config["name"]} is not implemented.')

    engine = trainer(model, optimizer, scalar, trainer_config['theta'])
    # --------------------------- Train/Validation -------------------------
    print('Training...')
    train_loss_epochs = []
    train_mape_epochs = []
    train_rmse_epochs = []
    val_loss_epochs = []
    val_mape_epochs = []
    val_rmse_epochs = []
    for epoch in range(trainer_config['epoch']):
        loss, mape, rmse = engine.train_one_epoch(data_loaders[0])
        train_loss_epochs.append(loss)
        train_mape_epochs.append(mape)
        train_rmse_epochs.append(rmse)

        print(f'epoch:{epoch}, train_loss(MAE):{train_loss_epochs[epoch]}, '
              f'train_MAPE:{train_mape_epochs[epoch]}, train_RMSE:{train_rmse_epochs[epoch]}')

        loss, mape, rmse = engine.eval_one_epoch(data_loaders[1])
        val_loss_epochs.append(loss)
        val_mape_epochs.append(mape)
        val_rmse_epochs.append(rmse)

        print(f'val_loss(MAE):{val_loss_epochs[epoch]}, '
              f'val_MAPE:{val_mape_epochs[epoch]}, val_RMSE:{val_rmse_epochs[epoch]}')

    torch.save(model.state_dict(), trainer_config['model_path'] + '/' + 'gwnet_pretrained' + '.pth')
    print('Model saved!')

    # --------------------------- Test -------------------------
    print('Testing...')
    test_loss, test_mape, test_rmse = engine.test(data_loaders[2])
    print(f'test_loss(MAE):{test_loss}, test_mape:{test_mape}, test_rmse:{test_rmse}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../config/config_graph_wavenet.yaml',
                        help='Config path')
    args = parser.parse_args()
    configs = load_config(args.config_path)
    main(configs)

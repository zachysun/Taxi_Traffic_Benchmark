import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.ma2gcn import MA2GCN
from trainer.trainer_ma2gcn import trainer
from utils import ProcessData, TransAdj, StandardScalar


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
    origin_adj = pd.read_csv(data_config['origin_adj_data_path'], header=None)
    origin_adj = torch.Tensor(origin_adj.values).to(device)
    io_matrix = np.load(data_config['io_adj_data_path'])
    io_adj = torch.Tensor(TransAdj(io_matrix)).to(device)

    # process io adj
    io_bound = data_config['io_matrix_interval'] * 60 / data_config['basic_interval']
    io_range = np.array(range(0, samples + 1, int(io_bound)))

    # get cur_io_adj for val and test
    val_start = train_data.shape[0]
    val_cur_io_adj = io_adj[np.digitize(val_start, io_range) - 2]
    test_start = val_start + val_data.shape[0]
    test_cur_io_adj = io_adj[np.digitize(test_start, io_range) - 2]

    # ------------------------ Model parameters getting ------------------------
    blocks_num = trainer_config['block_nums']
    kernel_size = trainer_config['kernel_size']
    in_tcn_dim_list = trainer_config['in_tcn_dim_list']
    adj_input_dim = origin_adj.shape[0]
    _, input_seq, nodes_num, features = train_x.shape
    # [Start]--get receptive field, dilation list, all_feature_dim, input_seq(padded)
    cur_dilation_list = [0] * blocks_num
    all_feature_dim = [0] * blocks_num
    cur_dilation_list[0] = 1
    receptive_field = 1
    additional_scope = kernel_size[1] - 1

    for i in range(blocks_num):
        receptive_field = receptive_field + additional_scope
        additional_scope *= 2

    new_input_seq = receptive_field
    all_feature_dim[0] = new_input_seq * features

    for j in range(1, blocks_num):
        cur_dilation_list[j] = 2 * cur_dilation_list[j - 1]
        all_feature_dim[j] = in_tcn_dim_list[j] * (
                new_input_seq - cur_dilation_list[j - 1] * (kernel_size[1] - 1))
        new_input_seq = new_input_seq - cur_dilation_list[j - 1] * (kernel_size[1] - 1)
    # [End]--get receptive field, dilation list, all_feature_dim, new_input_seq(padded)

    # ------------------------ Model setting ------------------------
    model = MA2GCN(trainer_config['block_nums'], trainer_config['K'], adj_input_dim, trainer_config['adj_hidden_dim'],
                   origin_adj, trainer_config['in_channel_list'], trainer_config['out_channel_list'],
                   trainer_config['in_tcn_dim_list'], trainer_config['out_tcn_dim_list'],
                   cur_dilation_list, trainer_config['kernel_size'], nodes_num, all_feature_dim,
                   receptive_field, trainer_config['drop_rate'], device)
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

    engine = trainer(model, optimizer, scalar, trainer_config['theta'], io_range, io_adj)
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

        loss, mape, rmse = engine.eval_one_epoch(data_loaders[1], val_cur_io_adj)
        val_loss_epochs.append(loss)
        val_mape_epochs.append(mape)
        val_rmse_epochs.append(rmse)

        print(f'val_loss(MAE):{val_loss_epochs[epoch]}, '
              f'val_MAPE:{val_mape_epochs[epoch]}, val_RMSE:{val_rmse_epochs[epoch]}')

    torch.save(model.state_dict(), trainer_config['model_path'] + '/' + 'ma2gcn_pretrained' + '.pth')
    print('Model saved!')

    # --------------------------- Test -------------------------
    print('Testing...')
    test_loss, test_mape, test_rmse = engine.test(data_loaders[2], test_cur_io_adj)
    print(f'test_loss(MAE):{test_loss}, test_mape:{test_mape}, test_rmse:{test_rmse}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../config/config_ma2gcn.yaml',
                        help='Config path')
    args = parser.parse_args()
    configs = load_config(args.config_path)
    main(configs)

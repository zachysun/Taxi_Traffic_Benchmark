from utils import ProcessData, masked_mae_np, masked_mape_np, masked_rmse_np
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import yaml
import argparse


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    data_config = config['dataset']
    feature_matrix = np.load(data_config['feature_matrix_data_path'])
    # shape -> (number of samples, sequence length, number of vertexes, features)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = ProcessData(feature_matrix,
                                                                                         data_config['input_seq'],
                                                                                         data_config['output_seq'],
                                                                                         data_config['split_rate'])

    data = test_data
    labels = test_labels
    reshaped_data = data.transpose(0, 2, 1, 3).reshape(-1, data.shape[1], data.shape[3])
    reshaped_data_1 = reshaped_data[..., 0]
    reshaped_data_2 = reshaped_data[..., 1]
    reshaped_labels = labels.transpose(0, 2, 1, 3).reshape(-1, labels.shape[1], labels.shape[3])
    reshaped_labels_1 = reshaped_labels[..., 0]
    reshaped_labels_2 = reshaped_labels[..., 1]

    preds = []
    for i in range(reshaped_data_1.shape[0]):
        node_feature_data = reshaped_data_1[i]
        model = ARIMA(node_feature_data, order=(1, 1, 1))
        model_fit = model.fit()
        y_pred = model_fit.forecast(steps=1)[0]
        preds.append(y_pred)

    preds = np.array(preds)
    preds = preds.reshape(len(preds), 1)
    loss1 = masked_mae_np(preds, reshaped_labels_1, 0.0)
    mape1 = masked_mape_np(preds, reshaped_labels_1, 0.0)
    rmse1 = masked_rmse_np(preds, reshaped_labels_1, 0.0)

    preds2 = []
    for i in range(reshaped_data_2.shape[0]):
        node_feature_data = reshaped_data_2[i]
        model = ARIMA(node_feature_data, order=(1, 1, 1))
        model_fit = model.fit()
        y_pred = model_fit.forecast(steps=1)[0]
        preds2.append(y_pred)

    preds2 = np.array(preds)
    preds2 = preds.reshape(len(preds), 1)
    loss2 = masked_mae_np(preds, reshaped_labels_2, 0.0)
    mape2 = masked_mape_np(preds, reshaped_labels_2, 0.0)
    rmse2 = masked_rmse_np(preds, reshaped_labels_2, 0.0)

    print((loss1 + loss2) / 2)
    print((mape1 + mape2) / 2)
    print((rmse1 + rmse2) / 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../config/config_arima.yaml',
                        help='Config path')
    args = parser.parse_args()
    configs = load_config(args.config_path)
    main(configs)

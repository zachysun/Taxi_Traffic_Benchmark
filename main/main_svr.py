from utils import ProcessData, masked_mae_np, masked_mape_np, masked_rmse_np, StandardScalar_np
import numpy as np
from sklearn.svm import SVR
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

    scalar = StandardScalar_np(train_data, axis=(0, 1, 2))
    data_x_sets = [train_data, val_data, test_data]
    data_y_sets = [train_labels, val_labels, test_labels]
    data_x_normed_sets = []
    data_y_real_sets = []
    data_y_normed_sets = []

    for data in data_x_sets:
        data_normed = scalar.transform(data)
        data_x_normed_sets.append(data_normed)
    train_x, val_x, test_x = data_x_normed_sets

    for data in data_y_sets:
        data_y_real_sets.append(data)
    train_real_y, val_real_y, test_real_y = data_y_real_sets

    for data in data_y_sets:
        data_normed = scalar.transform(data)
        data_y_normed_sets.append(data_normed)
    train_normed_y, val_normed_y, test_normed_y = data_y_normed_sets

    X_train = np.einsum('btnf->bnft', train_x)
    X_train = X_train.reshape(-1, X_train.shape[-1])
    y_train = np.reshape(train_normed_y, (-1))
    X_test = np.einsum('btnf->bnft', test_x)
    X_test = X_test.reshape(-1, X_test.shape[-1])
    y_test = np.reshape(test_real_y, (-1))
    b, t, n, f = test_real_y.shape

    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    y_pred = scalar.inverse_transform(y_pred.reshape(b, t, n, f)).reshape(-1)

    loss = masked_mae_np(y_pred, y_test, 0.0)
    mape = masked_mape_np(y_pred, y_test, 0.0)
    rmse = masked_rmse_np(y_pred, y_test, 0.0)

    print(loss)
    print(mape)
    print(rmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../config/config_svr.yaml',
                        help='Config path')
    args = parser.parse_args()
    configs = load_config(args.config_path)
    main(configs)

import numpy as np
import torch
from scipy.sparse.linalg import eigs


# ------------------------ Process data ------------------------

def ProcessData(data, in_seq_length, out_seq_length, rates):
    """
    :param data: raw data
    :param in_seq_length: the number of timestamps for prediction
    :param out_seq_length: the number of timestamps as labels
    :param rates: train/val/test
    :return: shape -> (number of samples, sequence length, number of vertexes, features)
    """
    # Set parameters
    features = data
    input_seq_length = in_seq_length
    output_seq_length = out_seq_length
    train_rate = rates[0]
    val_rate = rates[1]

    assert 10 - train_rate - val_rate == rates[2]

    # Get time sequence length
    data_length = features.shape[0]

    # Get index of train/val/test set
    train_length = int(train_rate / 10 * (data_length - input_seq_length - output_seq_length))
    val_length = int(val_rate / 10 * (data_length - input_seq_length - output_seq_length))

    train_start_idx = 0
    train_end_idx = train_length + 1

    val_start_idx = train_end_idx
    val_end_idx = val_start_idx + val_length + 1

    test_start_idx = val_end_idx
    test_end_idx = data_length - input_seq_length - output_seq_length + 1

    # Split dataset
    def SplitData(start_idx, end_idx):
        inputs = []
        labels = []
        for i in range(start_idx, end_idx):
            inputs.append(features[i:i + input_seq_length])
            labels.append(features[i + input_seq_length:i + input_seq_length + output_seq_length])
        return np.array(inputs), np.array(labels)

    train_data, train_labels = SplitData(train_start_idx, train_end_idx)
    val_data, val_labels = SplitData(val_start_idx, val_end_idx)
    test_data, test_labels = SplitData(test_start_idx, test_end_idx)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def TransAdj(io_matrix):
    # Set parameter
    io_matrix = io_matrix

    # Calculate weights
    num_trains = io_matrix[:, :, :, 0]
    travel_times = io_matrix[:, :, :, 1]
    weights = np.divide(num_trains, travel_times, out=np.zeros_like(num_trains), where=travel_times != 0)

    # Get adj_matrix
    adj_matrix = np.zeros((io_matrix.shape[0], io_matrix.shape[1], io_matrix.shape[1]))
    mask = np.sum(io_matrix, axis=-1) != 0
    adj_matrix[mask] = weights[mask]

    return adj_matrix


class StandardScalar:
    def __init__(self, data, device, axis=None):
        # Get mean, std, axis
        self.axis = axis
        self.mean = data.mean(axis=self.axis)
        self.std = data.std(axis=self.axis)
        self.device = device

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        self.std = torch.Tensor(self.std).to(self.device)
        self.mean = torch.Tensor(self.mean).to(self.device)
        return (data * self.std) + self.mean


class StandardScalar_np:
    def __init__(self, data, axis=None):
        # Get mean, std, axis
        self.axis = axis
        self.mean = data.mean(axis=self.axis)
        self.std = data.std(axis=self.axis)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# ------------------------ Model tools ------------------------

# Get L_tilde or L_sym-I or -D^-(1/2)*A*D^-(1/2)

def calculate_scaled_laplacian_torch(adj):
    """
    torch version
    """
    D = torch.diag(torch.sum(adj, dim=1))
    L = D - adj

    eigenvalues, _ = torch.linalg.eig(L)
    lambda_max = eigenvalues.real.max()

    return (2 * L) / lambda_max - torch.eye(adj.shape[0]).cuda()


def calculate_scaled_laplacian_np(adj):
    """
    numpy version
    """
    # if adj is tensor(cuda)
    adj = adj.cpu()
    adj = adj.detach().numpy()
    adj = np.array(adj)

    D = np.diag(np.sum(adj, axis=1))
    L = D - adj

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(adj.shape[0])


# Get chebyshev polynomials
def get_Tk(L_tilde, K):
    L_tilde = L_tilde.cpu()
    L_tilde = L_tilde.detach().numpy()
    L_tilde = np.array(L_tilde)

    T_ks = []
    N = L_tilde.shape[0]
    T_ks.append(np.identity(N))
    T_ks.append(L_tilde)

    for i in range(2, K):
        T_ks.append(2 * L_tilde * T_ks[i - 1] - T_ks[i - 2])

    return T_ks


# Get current io matrix
def IoMatrixSelect(io_adj, past_hours):
    number_segments = io_adj.shape[0]
    base_hour_interval = int(24 / number_segments)
    past_hours = past_hours

    return io_adj[int(past_hours / base_hour_interval)]


# ------------------------ Metrics ------------------------
# MASK MAE
def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_np(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds - labels)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


# MASK MAPE
def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape_np(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds - labels) / labels
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


# MASK MSE/RMSE
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return np.mean(loss)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))

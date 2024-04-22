import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LoadData(Dataset):
    def __init__(self, data_path, num_nodes, divide, time_interval, history_length, train_mode):

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_num = divide[0]
        self.test_num = divide[1]
        self.history_length = history_length
        self.time_interval = time_interval
        self.flow_norm, self.flow_data = self.pre_process_data(data=np.load(data_path[0]), norm_dim=1)
        self.graph = pd.read_csv('adj.csv', header=None).to_numpy()

    def __len__(self):

        if self.train_mode == "train":
            return self.train_num * self.time_interval
        elif self.train_mode == "test":
            return self.test_num * self.time_interval
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):

        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_num * self.time_interval
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)

        data_x = LoadData.to_tensor(data_x)
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data, history_length, index, train_mode):

        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index:end_index]
        data_y = data[:, end_index]

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):

        norm_base = LoadData.normalize_base(data, norm_dim)
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):

        max_data = np.max(data, norm_dim, keepdims=True)
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):

        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / (base + 1)

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):

        mid = min_data
        base = max_data - min_data

        recovered_data = data * (base + 1) + mid

        return recovered_data

    @staticmethod
    def to_tensor(data):
        data = torch.tensor(data, dtype=torch.float)
        return data

import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_data = LoadData(data_path=["/data.npy"], num_nodes=10, divide=[7000, 3000],
                          time_interval=10, history_length=10,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, drop_last=True, num_workers=1)

    test_data = LoadData(data_path=["/data.npy"], num_nodes=10, divide=[7000, 3000],
                         time_interval=10, history_length=10,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, drop_last=True, num_workers=1)
    max = train_data.flow_norm[0]
    mix = train_data.flow_norm[1]
    my_net = GCN_T(10, 10, 1)

    device = torch.device("cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    Epoch = 100
    train_losses_gcn, train_losses_gnn, train_losses_dnn, train_losses_lstm = [], [], [], []

    my_net.train()
    optimizer = optim.Adam(params=my_net.parameters(), lr=0.01)
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:
            predict_value = my_net(data)
            loss = criterion(predict_value, data["flow_y"])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        train_losses_gcn.append(epoch_loss / len(train_loader))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()

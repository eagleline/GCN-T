import torch
import torch.nn as nn

class GCN_T(nn.Module):
    def __init__(self, input_size, d_model, output_size):
        super(GCN_T, self).__init__()

        self.act = nn.ReLU()
        self.input_fc = nn.Linear(input_size, d_model)
        self.output_fc = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=10,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
            device='cpu'
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=10,
            dropout=0.1,
            dim_feedforward=4 * d_model,
            batch_first=True,
            device='cpu'
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=5)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_size)
        self.fc3 = nn.Linear(output_size, output_size)
        self.weight = nn.Parameter(torch.FloatTensor(input_size, output_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, data):
        device = 'cpu'
        x = data["flow_x"].to(device)
        graph_data = data["graph"].to(device)[0]
        graph_data = GCN_T.process_graph(graph_data)

        y = torch.matmul(graph_data, x)
        x = self.fc1(x) + y

        x = self.pos_emb(x)

        decoded = self.decoder(x)

        decoded = decoded.flatten(start_dim=2)

        out = self.fc2(decoded)
        out = self.act(out)
        out = self.fc3(out)

        return out

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)
        graph_data += matrix_i

        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.

        degree_matrix = torch.diag(degree_matrix)

        return torch.mm(degree_matrix, graph_data)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]



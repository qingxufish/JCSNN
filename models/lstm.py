import torch
import torch.nn as nn
from .BaseModel import BaseModel

class LSTM(BaseModel):
    def __init__(self, config):
        super(LSTM, self).__init__()
        input_dim = config.get('input_dim')
        hidden_dim = config.get('hidden_dim')
        num_layers = config.get('num_layers')

        self.lstm_encoder_u = nn.LSTM(input_dim*6, hidden_dim, num_layers, batch_first=True)
        self.lstm_encoder_v = nn.LSTM(input_dim * 6, hidden_dim, num_layers, batch_first=True)
        self.decoder_u = nn.Linear(hidden_dim, input_dim)
        self.decoder_v = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3)* x.size(4))
        output_u, _ = self.lstm_encoder_u(x)
        output_v, _ = self.lstm_encoder_v(x)

        output_u = torch.nn.functional.relu(output_u[:, -1, :])
        output_v = torch.nn.functional.relu(output_v[:, -1, :])

        u = self.decoder_u(output_u)
        v = self.decoder_u(output_v)
        return u, v

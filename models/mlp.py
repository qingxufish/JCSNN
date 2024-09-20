import torch
import torch.nn as nn
from .BaseModel import BaseModel

class Mlp(BaseModel):
    def __init__(self, config):
        super(Mlp, self).__init__()
        input_dim = config.get('input_dim')
        hidden_dim = config.get('hidden_dim')

        self.u_encoder = nn.Linear(input_dim * 6, hidden_dim)
        self.u_decoder = nn.Linear(hidden_dim, input_dim)
        self.v_encoder = nn.Linear(input_dim * 6, hidden_dim)
        self.v_decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, data):
        x = data.reshape(data.size(0),-1)
        # x = self.bn(x)
        u = self.u_encoder(x)
        u = torch.nn.functional.relu(u)
        u = self.u_decoder(u)

        v = self.v_encoder(x)
        v = torch.nn.functional.relu(v)
        v = self.v_decoder(v)

        return u ,v


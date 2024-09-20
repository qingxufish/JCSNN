import torch
import torch.nn as nn
from myUtils.positionEmbedding import PositionalEncoding
from .BaseModel import BaseModel

class ConvNet(BaseModel):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        kernel_size = config.get('kernel_size')
        channel_num = config.get('channel_num')
        n_head = config.get('n_head')
        num_layers = config.get('num_layers')

        self.Conv3DLayer = nn.Conv3d(1,32,kernel_size,1,0)

        self.pos_encoder = PositionalEncoding(d_model=channel_num)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=channel_num, nhead=n_head, dim_feedforward=channel_num)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)


    def forward(self, batch_data):
        x = self.Conv3DLayer(batch_data.view(batch_data.size(0),1,batch_data.size(1),batch_data.size(2),batch_data.size(3)))
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(x.size(0),x.size(1),-1)
        x = self.token_layer(x)

        x = self.transformer_encoder(x)
        x = x.reshape(-1, x.size(1) * x.size(2))
        x = torch.nn.functional.relu(x)

        u = self.u_layer(x)
        v = self.v_layer(x)

        return u, v
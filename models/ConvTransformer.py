import torch
import torch.nn as nn
from .BaseModel import BaseModel
from myUtils.attentionPackage import SelfAttention

class ConvTransformer(BaseModel):
    def __init__(self, config):
        super(ConvTransformer, self).__init__()
        kernel_size_u = config.get('kernel_size_u')
        kernel_size_v = config.get('kernel_size_v')
        channel_num_u = config.get('channel_num_u')
        channel_num_v = config.get('channel_num_v')
        n_head = config.get('n_head')
        num_layers = config.get('num_layers')
        image_size = config.get('image_size')
        downSamplingDim_u = config.get('downSamplingDim_u')
        downSamplingDim_v = config.get('downSamplingDim_v')

        self.Conv3DLayer_u = nn.Conv3d(1, channel_num_u, kernel_size_u, 1, 0)
        self.Conv3DLayer_v = nn.Conv3d(1, channel_num_v, kernel_size_v, 1, 0)

        self.downSamplingLayer_u = nn.Linear((image_size[0] - kernel_size_u[1] + 1) * (image_size[1] - kernel_size_u[2] + 1), downSamplingDim_u)
        self.downSamplingLayer_v = nn.Linear((image_size[0] - kernel_size_v[1] + 1) * (image_size[1] - kernel_size_v[2] + 1), downSamplingDim_v)

        # self.encoder_layer_u = nn.TransformerEncoderLayer(d_model=downSamplingDim_u, nhead=n_head, dropout=0.0, batch_first=True)
        # self.encoder_layer_v = nn.TransformerEncoderLayer(d_model=downSamplingDim_v, nhead=n_head, dropout=0.0, batch_first=True)
        #
        # self.transformer_encoder_u = nn.TransformerEncoder(self.encoder_layer_u, num_layers=num_layers)
        # self.transformer_encoder_v = nn.TransformerEncoder(self.encoder_layer_v, num_layers=num_layers)

        self.transformer_encoder_u = SelfAttention(1, downSamplingDim_u, downSamplingDim_u, 0.0)
        self.transformer_encoder_v = SelfAttention(1, downSamplingDim_v, downSamplingDim_v, 0.0)

        self.u_layer = nn.Linear(downSamplingDim_u*channel_num_u, image_size[0]*image_size[1])
        self.v_layer = nn.Linear(downSamplingDim_v*channel_num_v, image_size[0]*image_size[1])

    def forward(self, batch_data):
        u = self.Conv3DLayer_u(batch_data.view(batch_data.size(0), 1, batch_data.size(1), batch_data.size(2), batch_data.size(3)))
        v = self.Conv3DLayer_v(batch_data.view(batch_data.size(0), 1, batch_data.size(1), batch_data.size(2), batch_data.size(3)))

        u = u.reshape(u.size(0),u.size(1),-1)
        v = v.reshape(v.size(0), v.size(1), -1)

        u = torch.nn.functional.relu(u)
        v = torch.nn.functional.relu(v)

        u = self.downSamplingLayer_u(u)
        v = self.downSamplingLayer_v(v)

        u = self.transformer_encoder_u(u)
        v = self.transformer_encoder_v(v)

        u = u.reshape(-1, u.size(1) * u.size(2))
        v = v.reshape(-1, v.size(1) * v.size(2))

        u = torch.nn.functional.relu(u)
        v = torch.nn.functional.relu(v)

        u = self.u_layer(u)
        v = self.v_layer(v)

        return u, v

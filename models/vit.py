import torch
import torch.nn as nn
from myUtils.positionEmbedding import PositionalEncoding
from myUtils.attentionPackage import SelfAttention
from .BaseModel import BaseModel

class VIT(BaseModel):
    def __init__(self, config):
        super(VIT, self).__init__()
        n_head = config.get('n_head')
        num_layers = config.get('num_layers')
        patch_size = config.get('patch_size')
        image_size = config.get('image_size')
        token_dim = config.get('token_dim')
        self.windowSize = patch_size

        self.pos_encoder = PositionalEncoding(d_model=token_dim)
        self.encoder_layer_u = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_head,dropout=0.0, batch_first=True)
        self.encoder_layer_v = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_head, dropout=0.0, batch_first=True)
        self.transformer_encoder_u = nn.TransformerEncoder(self.encoder_layer_u, num_layers=num_layers)
        self.transformer_encoder_v = nn.TransformerEncoder(self.encoder_layer_v, num_layers=num_layers)

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.token_layer_u = nn.Sequential(nn.Linear(patch_size[0]*patch_size[1]*6, token_dim*2),
                                         nn.ReLU(),
                                         nn.Linear(token_dim*2, token_dim)
                                         ) #对patch进行tokenize的mlp

        self.token_layer_v = nn.Sequential(nn.Linear(patch_size[0]*patch_size[1]*6, token_dim*2),
                                         nn.ReLU(),
                                         nn.Linear(token_dim*2, token_dim)
                                         ) #对patch进行tokenize的mlp

        self.u_layer = nn.Linear(num_patches*token_dim, image_size[0]*image_size[1])
        self.v_layer = nn.Linear(num_patches*token_dim, image_size[0]*image_size[1])

    def splitFigure(self, batchFigure, windowSize):
        B ,C ,H, W = batchFigure.shape
        x = batchFigure.view(B ,C , H // windowSize[0], windowSize[0], W // windowSize[1], windowSize[1])
        windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B ,C , -1, windowSize[0]*windowSize[1])
        return windows

    def forward(self, batch_data):
        u = self.splitFigure(batch_data, self.windowSize)
        v = self.splitFigure(batch_data, self.windowSize)

        u = u.permute(0,2,1,3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        u = u.view(u.size(0),u.size(1),-1)
        v = v.view(v.size(0), v.size(1), -1)

        u = self.token_layer_u(u)
        v = self.token_layer_v(v)

        u = self.transformer_encoder_u(u)
        v = self.transformer_encoder_v(v)

        u = u.reshape(-1, u.size(1) * u.size(2))
        v = v.reshape(-1, v.size(1) * v.size(2))
        u = torch.nn.functional.relu(u)
        v = torch.nn.functional.relu(v)

        u = self.u_layer(u)
        v = self.v_layer(v)

        return u, v

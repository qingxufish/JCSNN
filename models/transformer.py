import torch.nn as nn
from myUtils.positionEmbedding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, nhead, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=int(dim_feedforward/2))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder1 = nn.Linear(dim_feedforward, dim_feedforward*3)
        self.decoder2 = nn.Linear(dim_feedforward*3, int(dim_feedforward/2))

    def forward(self, src):
        src = self.pos_encoder(src).view(src.size(0),src.size(1)+1,-1)
        output = self.transformer_encoder(src)
        output = self.decoder2(self.decoder1(output[:,-1,:]))
        return output
import torch.nn as nn
from einops.layers.torch import Rearrange

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

class mixMlp(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, out_channel, expansion_factor=4, dropout=0.):
        super(mixMlp, self).__init__()
        self.net = self.MLPMixer(image_size=image_size, patch_size=patch_size, dim=dim,
                                 depth=depth, expansion_factor=expansion_factor, dropout=dropout)
        self.out_channel = out_channel

    def MLPMixer(self, image_size, patch_size, dim, depth, expansion_factor=4, dropout=0.):
        assert (image_size[0] % patch_size[0]) + (image_size[1] % patch_size[1]) == 0, 'image must be divisible by patch size'
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.u_layer = nn.Linear(num_patches*dim//2, image_size[0]*image_size[1])
        self.v_layer = nn.Linear(num_patches*dim//2, image_size[0]*image_size[1])

        return nn.Sequential(
            # 1. 将图片拆成多个patches
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size[0], p2=patch_size[1]),
            # 2. 用一个全连接网络对所有patch进行处理，提取出tokens, 类似于归一化
            nn.Linear((patch_size[0] * patch_size[1]) * 5, dim),
            # 3. 加入非线性层，使之实现通道混合
            nn.LeakyReLU(0.1),
            # 4. 经过N个Mixer层，混合提炼特征信息
            *[nn.Sequential(
                PreNormResidual(dim, nn.Sequential(
                    Rearrange('b n c -> b c n'),
                    FeedForward(num_patches, expansion_factor, dropout),
                    Rearrange('b c n -> b n c'),
                )),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout))
            ) for _ in range(depth)]
        )

    def forward(self, batch_data):
        x = self.net(batch_data)
        x = x.view(x.shape[0], -1)
        u = self.u_layer(x[:, 0:x.shape[1]//2])
        v = self.v_layer(x[:, x.shape[1]//2:])
        return u ,v

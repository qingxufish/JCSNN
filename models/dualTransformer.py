import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from einops.layers.torch import Rearrange
import numpy as np
import matplotlib.pyplot as plt
import pickle

import netCDF4

class ImageDataset(Dataset):
    def __init__(self, nc_data):
        self.nc_data = nc_data.to(torch.float32)


    def __len__(self):
        return len(self.nc_data)

    def __getitem__(self, idx):
        try:
            data = self.nc_data[idx,:]
            label = self.nc_data[idx+1,:] # 只截取表层uv数据，按照时间进行索引
        except:
            data = self.nc_data[idx-1,:]
            label = self.nc_data[idx,:] # 只截取表层uv数据，按照时间进行索引
        return data, label

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].repeat(x.size(0),1,1)
        return x

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

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        train_data =  data[0]
        labels = data[1]
        train_data, labels = train_data.to(device), labels.to(device)
        optimizer.zero_grad()
        u, v = model(train_data)
        loss = (criterion(u, labels[:,0,:].reshape(labels.size(0),-1)) + criterion(v, labels[:,1,:].reshape(labels.size(0), -1)))/2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"loss:{loss.item():.4f}")
    return running_loss / (i + 1)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        train_data =  data[0]
        labels = data[1]
        train_data, labels = train_data.to(device), labels.to(device)
        u, v = model(train_data)
        loss = (criterion(u, labels[:,0,:].reshape(labels.size(0),-1)) + criterion(v, labels[:,1,:].reshape(labels.size(0), -1)))/2
        running_loss += loss.item()
    return running_loss / (i + 1)

@torch.no_grad()
def test(model, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        train_data =  data[0]
        labels = data[1]
        train_data, labels = train_data.to(device), labels.to(device)
        u, v = model(train_data)
        loss = (criterion(u, labels[:,0,:].reshape(labels.size(0),-1)) + criterion(v, labels[:,1,:].reshape(labels.size(0), -1)))/2
        running_loss += loss.item()

        x = np.linspace(1, 40, 40)
        y = np.linspace(1, 30, 30)
        u = u.reshape(-1, 30, 40)
        v = v.reshape(-1, 30, 40)
        u_l = labels[:, 0, :]
        v_l = labels[:, 1, :]
        X, Y = np.meshgrid(x, y)
        for index in range(len(u)):
            fig, ax = plt.subplots()
            ax.quiver(X, Y, u[index, 0::1, 0::1].to('cpu').numpy(), v[index, 0::1, 0::1].to('cpu').numpy(), units='width', color='red')
            ax.quiver(X, Y, u_l[index, 0::1, 0::1].to('cpu').numpy(), v_l[index, 0::1, 0::1].to('cpu').numpy(),
                      units='width', color='blue')
            plt.savefig(f'./pic/{index}.png')
            plt.close()


    return running_loss / (i + 1)

def main():
    # load nc data
    file_list = ['data_u.nc', 'data_v.nc', 'data_taux.nc', 'data_tauy.nc', 'data_ssh.nc']
    nc_data = []
    for file_name in file_list:
        variable_name_current = file_name[5:-3]
        temp_data = netCDF4.Dataset(f'data/{file_name}', 'r').variables[variable_name_current][:].filled(0)
        if variable_name_current in ['u','v']:
            temp_data = temp_data[:,0,:] # 只获取表层数据
        temp_data = np.interp(temp_data, (temp_data.min(), temp_data.max()), (0, 1))
        nc_data.append(temp_data)

    nc_data = torch.tensor(nc_data)
    nc_data = nc_data.transpose(1,0)
    # Hyperparameters
    image_size = (nc_data.shape[2],nc_data.shape[3])  # 输入图像的尺寸
    # 定义超参数网格
    # param_grid = {
    # 'patch_size' : [(2, 4), (3, 4), (5, 4), (6, 4), (10, 4), (15, 4),
    #                 (2, 2), (3, 2), (5, 2), (6, 2), (10, 2), (15, 2),
    #                 (2, 8), (3, 8), (5, 8), (6, 8), (10, 8), (15, 8),
    #                 (2, 10), (3, 10), (5, 10), (6, 10), (10, 10), (15, 10),
    #                 (2, 20), (3, 20), (5, 20), (6, 20), (10, 20), (15, 20)
    #                 ],  # 小图的尺寸
    # 'dim' : list(range(30,330,30)),
    # 'depth' : list(range(1,7,1)),  # mix层的层数
    # 'out_channel' : [0],  # 输出预测通道
    # 'expansion_factor' : list(range(1,10,1)),
    # 'batch_size' : list(range(2,64,2)),
    # 'learning_rate' : [0.000005,0.00005,0.0005,0.005,0.05]
    # }
    patch_size = (3,4) # 小图的尺寸
    dim = 128  # 小图
    depth = 2  # mix层的层数
    out_channel = 0 # 输出预测通道
    expansion_factor = 4
    dropout = 0.0
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.00005
    # 早停法
    min_loss = float('inf')
    patience = 10
    wait = 0
    # data loader
    time_length = nc_data.shape[0]
    train_data = ImageDataset(nc_data[0:int(time_length / 10 * 6), :])
    valid_data = ImageDataset(nc_data[int(time_length / 10 * 6):int(time_length / 10 * 8), :])
    test_data = ImageDataset(nc_data[int(time_length / 10 * 8):, :])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mixMlp(image_size=image_size, patch_size=patch_size, dim=dim,
                   depth=depth, expansion_factor=expansion_factor, dropout=dropout, out_channel=out_channel).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型的参数数量为：{num_params}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, criterion, optimizer, device)
        valid_loss = evaluate(model, valid_dataloader, criterion, device)
        # 如果验证集上的均方误差减小，则更新最小均方误差和最佳轮次
        if valid_loss < min_loss:
            min_loss = valid_loss
            wait = 0
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')
            print(f'validLoss: {valid_loss:.4f}')
        else:
            wait += 1
            if wait >= patience:
                break
    test_loss = test(model, test_dataloader, criterion, device)
    print(f'testLoss: {test_loss:.4f}')
    torch.save(model.state_dict(), '../model_result/mixTransfoomer_model.pth')

def forcast():
    # load nc data
    file_list = ['data_u.nc', 'data_v.nc', 'data_taux.nc', 'data_tauy.nc', 'data_ssh.nc']
    nc_data = []
    for file_name in file_list:
        variable_name_current = file_name[5:-3]
        temp_data = netCDF4.Dataset(f'data/{file_name}', 'r').variables[variable_name_current][:].filled(0)
        if variable_name_current in ['u', 'v']:
            temp_data = temp_data[:, 0, :]  # 只获取表层数据
        temp_data = np.interp(temp_data, (temp_data.min(), temp_data.max()), (0, 1))
        nc_data.append(temp_data)

    nc_data = torch.tensor(nc_data)
    nc_data = nc_data.transpose(1, 0)
    # Hyperparameters
    image_size = (nc_data.shape[2], nc_data.shape[3])  # 输入图像的尺寸
    patch_size = (5, 5)  # 小图的尺寸
    dim = 60  # 小图
    depth = 6  # mix层的层数
    out_channel = 0  # 输出预测通道
    expansion_factor = 4
    dropout = 0.0
    batch_size = 32
    # data loader
    time_length = nc_data.shape[0]
    train_data = ImageDataset(nc_data[0:int(time_length / 10 * 6), :])
    valid_data = ImageDataset(nc_data[int(time_length / 10 * 6):int(time_length / 10 * 8), :])
    test_data = ImageDataset(nc_data[int(time_length / 10 * 8):, :])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mixMlp(image_size=image_size, patch_size=patch_size, dim=dim,
                   depth=depth, expansion_factor=expansion_factor, dropout=dropout, out_channel=out_channel).to(device)
    saved_model = torch.load('../model_result/mixTransfoomer_model.pth')
    model.load_state_dict(saved_model)
    model()


if __name__ == '__main__':
    main()


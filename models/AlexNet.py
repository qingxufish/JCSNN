import torch
import torch.nn as nn
from .BaseModel import BaseModel
# 定义 AlexNet模型的类
class AlexNet(BaseModel):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        kernel_size = config.get('kernel_size')
        channel_num = config.get('channel_num')
        n_head = config.get('n_head')
        num_layers = config.get('num_layers')

        self.Conv3DLayer1 = nn.Conv3d(1, 32, kernel_size, 1, 0)
        self.Conv3DLayer2 = nn.Conv2d(32, 64, kernel_size, 1, 0)
        self.Conv3DLayer3 = nn.Conv2d(1, 32, kernel_size, 1, 0)
        self.conv1 = nn.Conv2d(3, 64, 7, 2)  # 3个输入通道,64个输出通道,7x7的卷积核,2x2的定位卷积
        self.conv2 = nn.Conv2d(64, 64, 5, 2)  # 64个输入通道,64个输出通道,5x5的卷积核,2x2的定位卷积
        self.conv3 = nn.Conv2d(64, 128, 3, 1)  # 64个输入通道,128个输出通道,3x3的卷积核,1x1的定位卷积
        self.pool = nn.MaxPool2d(2, 2)  # 2x2的最大池化层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 512个输入节点,1024个输出节点
        self.fc2 = nn.Linear(512, num_classes)  # num_classes个输出节点

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(x, dim=-1)
        return x
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from myUtils.dataSet import Dataset
from myUtils.train_valid_test import train, evaluate, test
from myUtils.data_analyse import data_analyse
from myUtils.combinations import combination
import argparse
import os
import json
from models import *
from tqdm import tqdm

def load_json_config(config_path):
    logging.info(' Loading configuration '.center(100, '-'))
    if not os.path.exists(config_path):
        logging.warning(f'File {config_path} does not exist, empty list is returned.')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['GPU']:
        config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        config['device'] = torch.device('cpu')
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/mlp.json', type=str, required=False,
                        help='选择模型参数')

    args = parser.parse_args()
    config = load_json_config(args.model_config)
    data_plot = data_analyse()
    # 读取指定文件
    data_set = Dataset("kuroShio", config.get('timeStep'))
    train_dataloader = DataLoader(data_set.data["train"], batch_size=config.get('batch_size'), shuffle=True)
    valid_dataloader = DataLoader(data_set.data["valid"], batch_size=config.get('batch_size'), shuffle=True)
    test_dataloader = DataLoader(data_set.data["test"], batch_size=config.get('batch_size'), shuffle=False)
    # 读取迁移学习文件
    data_set.read_transfer_nc_file([130,150],[0,15])
    transfer_test_dataloader = DataLoader(data_set.data["transfer"], batch_size=config.get('batch_size'), shuffle=False)

    # 计算每个点的权重系数，输出sigma
    data_set_sigma = Dataset("kuroShio")
    uv_var = torch.var(data_set_sigma.data["train"].nc_data[:, 0:2, :, :], 0)
    sigma = torch.cat(
        [uv_var[0:1, :, :] / torch.sum(uv_var, dim=[1, 2])[0], uv_var[1:2, :, :] / torch.sum(uv_var, dim=[1, 2])[1]],
        dim=0)
    # 读取参数
    parameters_collections = combination(config)
    for parameter_setting in tqdm(parameters_collections):

        # 超参数
        num_epochs = parameter_setting.get('num_epochs')
        learning_rate = parameter_setting.get('lr')

        # 初始化模型、优化器
        model, device = init_model(parameter_setting)
        criterion = nn.MSELoss(reduction='none')
        # criterion = nn.MSELoss()
        # sigma = None
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        min_loss = float('inf')
        patience = parameter_setting.get('patience')
        wait = 0

        for epoch in range(num_epochs):
            loss = train(model, train_dataloader, criterion, optimizer, device, sigma)
            valid_loss = evaluate(model, valid_dataloader, criterion, device, sigma)
            # 如果验证集上的均方误差减小，则更新最小均方误差和最佳轮次
            if valid_loss < min_loss:
                min_loss = valid_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        DMSE_test_loss,  u_collection, lu_collection, v_collection, lv_collection = test(model, test_dataloader, criterion, device, plotPath=None ,sigma=sigma ,datasetClass=data_set)
        MSE_test_loss, _, _, _, _ = test(model, test_dataloader, nn.MSELoss(),device, plotPath=None, sigma=None,datasetClass=data_set)
        transfer_MSE_test_loss, _, _, _, _ = test(model, transfer_test_dataloader, nn.MSELoss(), device, plotPath=None, sigma=None,
                                         datasetClass=data_set)
        print(f'DMSE为：{DMSE_test_loss}\nMSE为：{MSE_test_loss}\n迁移MSE为{transfer_MSE_test_loss}')
        # 输出参数
        data_plot.refreshData(DMSE_test_loss, MSE_test_loss, transfer_MSE_test_loss, parameter_setting)
        data_plot.output_parameters()
        # 输出uv
        # data_plot.output_to_excel([u_collection, lu_collection, v_collection, lv_collection], 'local')
    # print(f'泛化loss为:{sum(data_plot.local_loss)/len(data_plot.local_loss)}\n迁移loss为:{sum(data_plot.transfer_loss)/len(data_plot.transfer_loss)}')

if __name__ == '__main__':
    main()
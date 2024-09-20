from torch.utils.data import Dataset
import torch
import logging
import os
import netCDF4
import numpy as np
# 数据类，存储和获取数据
class ImageDataset(Dataset):
    def __init__(self, nc_data, timeStep=None):
        self.timeStep = timeStep
        if timeStep is None:
            self.nc_data = nc_data.to(torch.float32)
        else:
            nc_data = nc_data.to(torch.float32)
            self.nc_data = [nc_data[ind:ind+timeStep] for ind in range(len(nc_data)-timeStep)]

    def __len__(self):
        return len(self.nc_data)

    def __getitem__(self, idx):
        if self.timeStep is None:
            try:
                data = self.nc_data[idx,:]
                label = self.nc_data[idx+1,:] # 只截取表层uv数据，按照时间进行索引
            except:
                data = self.nc_data[idx-1,:]
                label = self.nc_data[idx,:] # 只截取表层uv数据，按照时间进行索引
        else:
            try:
                data = self.nc_data[idx][0:self.timeStep-1,:]
                label = self.nc_data[idx][self.timeStep-1:self.timeStep,:] # 只截取表层uv数据，按照时间进行索引
            except:
                data = self.nc_data[idx][0:self.timeStep-1,:]
                label = self.nc_data[idx][self.timeStep-1:self.timeStep,:] # 只截取表层uv数据，按照时间进行索引
        return data, label


class Dataset:
    def __init__(self, dataset, timeStep=None):
        self.timeStep = timeStep
        self.path = os.getcwd()
        self.name = dataset
        self.nc_data = []
        self.transfer_nc_data = []
        self.nc_u_max = None
        self.nc_u_min = None
        self.nc_v_max = None
        self.nc_v_min = None
        self.transfer_u_max = None
        self.transfer_u_min = None
        self.transfer_v_max = None
        self.transfer_v_min = None
        self.mask_array = None
        self.read_nc_file()
        self.time_length = self.nc_data.shape[0]
        try:
            self.data = {
                'train': self.read_train(self.timeStep),
                'valid': self.read_valid(self.timeStep),
                'test': self.read_test(self.timeStep)
            }
        except IOError:
            print(f"在{os.path.join(self.path, 'data', self.name)}中未找到相应文件")

    def read_nc_file(self): # 读取NC文件
        file_list = ['data_u.nc', 'data_v.nc', 'data_taux.nc', 'data_tauy.nc', 'data_ssh.nc']
        nc_data = []
        for file_name in file_list:
            variable_name_current = file_name[5:-3]
            temp_data = netCDF4.Dataset(f'data/{file_name}', 'r').variables[variable_name_current][:].filled(0)
            if variable_name_current in ['u', 'v']:
                temp_data = temp_data[:, 0, :]  # 只获取表层数据
                if variable_name_current == 'u':
                    self.nc_u_min = temp_data.min()
                    self.nc_u_max = temp_data.max()
                    self.mask_array = netCDF4.Dataset(f'data/{file_name}', 'r').variables[variable_name_current][0, 0, :].mask
                if variable_name_current == 'v':
                    self.nc_v_min = temp_data.min()
                    self.nc_v_max = temp_data.max()
                    self.mask_array = netCDF4.Dataset(f'data/{file_name}', 'r').variables[variable_name_current][0, 0, :].mask
            temp_data = np.interp(temp_data, (temp_data.min(), temp_data.max()), (-1, 1))
            nc_data.append(temp_data)
        lat_mesh = np.array(
            [[[item] * 40 for item in np.sin(netCDF4.Dataset(f'data/{file_name}', 'r').variables['latitude'][:] / 180 * 3.1415)]]*432
        )
        nc_data.append(lat_mesh)
        self.nc_data = torch.tensor(nc_data)
        self.nc_data = self.nc_data.transpose(1, 0)
        print("nc数据读取完毕")

    def read_transfer_nc_file(self,lon,lat): # 读取迁移学习NC文件
        file_list = ['data_u.nc', 'data_v.nc', 'data_taux.nc', 'data_tauy.nc', 'data_ssh.nc']
        nc_data = []
        for file_name in file_list:
            longitude = netCDF4.Dataset(f'data/pacific_sea/{file_name}', 'r').variables['longitude'][:]
            latitude = netCDF4.Dataset(f'data/pacific_sea/{file_name}', 'r').variables['latitude'][:]
            lon_start = abs(np.array(longitude) - lon[0]).argmin()
            lon_end = abs(np.array(longitude) - lon[1]).argmin()
            lat_start = abs(np.array(latitude) - lat[0]).argmin()
            lat_end = abs(np.array(latitude) - lat[1]).argmin()
            variable_name_current = file_name[5:-3]
            if variable_name_current in ['u', 'v']:
                temp_data = netCDF4.Dataset(f'data/pacific_sea/{file_name}', 'r').variables[variable_name_current][:, 0, lat_start:lat_end,lon_start:lon_end].filled(0)  # 只获取表层数据
                if variable_name_current == 'u':
                    self.transfer_u_min = temp_data.min()
                    self.transfer_u_max = temp_data.max()
                if variable_name_current == 'v':
                    self.transfer_v_min = temp_data.min()
                    self.transfer_v_max = temp_data.max()
            else:
                temp_data = netCDF4.Dataset(f'data/pacific_sea/{file_name}', 'r').variables[variable_name_current][:, lat_start:lat_end, lon_start:lon_end].filled(0)  # 只获取表层数据
            temp_data = np.interp(temp_data, (temp_data.min(), temp_data.max()), (-1, 1))
            nc_data.append(temp_data)
        lat_mesh = np.array(
            [[[item] * (lon_end-lon_start) for item in np.sin(netCDF4.Dataset(f'data/pacific_sea/{file_name}', 'r').variables['latitude'][lat_start:lat_end] / 180 * 3.1415)]]*432
        )
        nc_data.append(lat_mesh)
        self.transfer_nc_data = torch.tensor(nc_data)
        self.transfer_nc_data = self.transfer_nc_data.transpose(1, 0)
        if self.timeStep is None:
            self.data.update({'transfer':ImageDataset(self.transfer_nc_data[:, :])})
        else:
            self.data.update({'transfer': ImageDataset(self.transfer_nc_data[:, :], self.timeStep)})
        print("迁移学习nc数据读取完毕")

    def read_train(self,timeStep=None):
        if timeStep is None:
            logging.info(' Loading training data '.center(100, '-'))
            return ImageDataset(self.nc_data[0:int(self.time_length / 10 * 6), :])
        else:
            logging.info(' Loading training data '.center(100, '-'))
            return ImageDataset(self.nc_data[0:int(self.time_length / 10 * 6), :], timeStep)

    def read_valid(self,timeStep=None):
        if timeStep is None:
            logging.info(' Loading validation data '.center(100, '-'))
            return ImageDataset(self.nc_data[int(self.time_length / 10 * 6):int(self.time_length / 10 * 8), :])
        else:
            logging.info(' Loading validation data '.center(100, '-'))
            return ImageDataset(self.nc_data[int(self.time_length / 10 * 6):int(self.time_length / 10 * 8), :], timeStep)

    def read_test(self,timeStep=None):
        if timeStep is None:
            logging.info(' Loading testing data '.center(100, '-'))
            return ImageDataset(self.nc_data[int(self.time_length / 10 * 8):, :])
        else:
            logging.info(' Loading testing data '.center(100, '-'))
            return ImageDataset(self.nc_data[int(self.time_length / 10 * 8):, :], timeStep)

    def nc_reduce(self, data, uv_flag:str):
        result = None
        if uv_flag == 'u':
            result = (data+(self.nc_u_max + self.nc_u_min)/(self.nc_u_max - self.nc_u_min)) * (self.nc_u_max - self.nc_u_min) /2
        if uv_flag == 'v':
            result = (data+(self.nc_v_max + self.nc_v_min)/(self.nc_v_max - self.nc_v_min)) * (self.nc_v_max - self.nc_v_min) /2
        return result

    def transfer_reduce(self, data, uv_flag:str):
        result = None
        if uv_flag == 'u':
            result = (data+(self.transfer_u_max + self.transfer_u_min)/(self.transfer_u_max - self.transfer_u_min)) * (self.transfer_u_max - self.transfer_u_min) /2
        if uv_flag == 'v':
            result = (data+(self.transfer_v_max + self.transfer_v_min)/(self.transfer_v_max - self.transfer_v_min)) * (self.transfer_v_max - self.transfer_v_min) /2
        return result





import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from myUtils.continue_loss import continue_loss

def train(model, dataloader, criterion, optimizer, device, sigma=None):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        train_data = data[0]
        labels = data[1]
        train_data, labels = train_data.to(device), labels.to(device)
        optimizer.zero_grad()
        u, v = model(train_data)
        loss2 = 0
        # if len(labels.size()) < 5:
        #     loss2 = continue_loss(u.reshape(labels[: ,0, :].size()[0],labels[: ,0, :].size()[1],labels[: ,0, :].size()[2]),
        #                               v.reshape(labels[: ,0, :].size()[0],labels[: ,1, :].size()[1],labels[: ,0, :].size()[2]))
        # else:
        #     loss2 = continue_loss(u.reshape(labels[:,0 ,0, :].size()[0],labels[:,0 ,0, :].size()[1],labels[:,0 ,0, :].size()[2]),
        #                               v.reshape(labels[:,0 ,0, :].size()[0],labels[:,0, 1, :].size()[1],labels[:,0, 1, :].size()[2]))
        if len(labels.size()) < 5:
            if sigma is not None:
                loss = (torch.mean(torch.matmul(criterion(u, labels[: ,0, :].reshape(labels.size(0), -1)),
                                                sigma[0, :, :].to(device).view(-1, 1)))
                        + torch.mean(torch.matmul(criterion(v, labels[: , 1, :].reshape(labels.size(0), -1)),
                                                  sigma[1, :, :].to(device).view(-1, 1)))) / 2 + loss2
            else:
                loss = (criterion(u, labels[: ,0, :].reshape(labels.size(0), -1)) + criterion(v, labels[: , 1, :].reshape(
                    labels.size(0), -1))) / 2 + loss2
        else:
            if sigma is not None:
                loss = (torch.mean(torch.matmul(criterion(u, labels[:,0 ,0, :].reshape(labels.size(0), -1)),
                                                sigma[0, :, :].to(device).view(-1, 1)))
                        + torch.mean(torch.matmul(criterion(v, labels[:,0, 1, :].reshape(labels.size(0), -1)),
                                                  sigma[1, :, :].to(device).view(-1, 1)))) / 2 + loss2
            else:
                loss = (criterion(u, labels[:,0 ,0, :].reshape(labels.size(0), -1)) + criterion(v, labels[:,0 , 1, :].reshape(
                    labels.size(0), -1))) / 2 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # print(f"loss:{loss.item():.4f}")
    return running_loss / (i + 1)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, sigma=None):
    model.eval()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        train_data =  data[0]
        labels = data[1]
        train_data, labels = train_data.to(device), labels.to(device)
        u, v = model(train_data)
        if len(labels.size()) < 5:
            if sigma is not None:
                loss = (torch.mean(torch.matmul(criterion(u, labels[:, 0, :].reshape(labels.size(0), -1)),
                                                sigma[0, :, :].to(device).view(-1, 1)))
                        + torch.mean(torch.matmul(criterion(v, labels[:, 1, :].reshape(labels.size(0), -1)),
                                                  sigma[1, :, :].to(device).view(-1, 1)))) / 2
            else:
                loss = (criterion(u, labels[:, 0, :].reshape(labels.size(0), -1)) + criterion(v,
                                                                                              labels[:, 1, :].reshape(
                                                                                                  labels.size(0),
                                                                                                  -1))) / 2
        else:
            if sigma is not None:
                loss = (torch.mean(torch.matmul(criterion(u, labels[:, 0, 0, :].reshape(labels.size(0), -1)),
                                                sigma[0, :, :].to(device).view(-1, 1)))
                        + torch.mean(torch.matmul(criterion(v, labels[:, 0, 1, :].reshape(labels.size(0), -1)),
                                                  sigma[1, :, :].to(device).view(-1, 1)))) / 2
            else:
                loss = (criterion(u, labels[:, 0, 0, :].reshape(labels.size(0), -1)) + criterion(v, labels[:, 0, 1,
                                                                                                    :].reshape(
                    labels.size(0), -1))) / 2
        running_loss += loss.item()
    return running_loss / (i + 1)

@torch.no_grad()
def test(model, dataloader, criterion, device,plotPath=None, sigma=None, datasetClass=None):
    model.eval()
    running_loss = 0.0
    u_collection = []
    v_collection = []
    lu_collection = []
    lv_collection = []

    u_shape_collection = torch.tensor([],device=device)
    v_shape_collection = torch.tensor([],device=device)
    lu_shape_collection = torch.tensor([],device=device)
    lv_shape_collection = torch.tensor([],device=device)

    pic_index = 0
    for i, data in enumerate(dataloader):
        train_data =  data[0]
        labels = data[1]
        train_data, labels = train_data.to(device), labels.to(device)
        u, v = model(train_data)
        if len(labels.size()) < 5:
            if sigma is not None:
                loss = (torch.mean(torch.matmul(criterion(u, labels[:, 0, :].reshape(labels.size(0), -1)),
                                                sigma[0, :, :].to(device).view(-1, 1)))
                        + torch.mean(torch.matmul(criterion(v, labels[:, 1, :].reshape(labels.size(0), -1)),
                                                  sigma[1, :, :].to(device).view(-1, 1)))) / 2
            else:
                loss = (criterion(u, labels[:, 0, :].reshape(labels.size(0), -1)) + criterion(v,
                                                                                              labels[:, 1, :].reshape(
                                                                                                  labels.size(0),
                                                                                                  -1))) / 2
        else:
            if sigma is not None:
                loss = (torch.mean(torch.matmul(criterion(u, labels[:, 0, 0, :].reshape(labels.size(0), -1)),
                                                sigma[0, :, :].to(device).view(-1, 1)))
                        + torch.mean(torch.matmul(criterion(v, labels[:, 0, 1, :].reshape(labels.size(0), -1)),
                                                  sigma[1, :, :].to(device).view(-1, 1)))) / 2
            else:
                loss = (criterion(u, labels[:, 0, 0, :].reshape(labels.size(0), -1)) + criterion(v, labels[:, 0, 1,
                                                                                                    :].reshape(
                    labels.size(0), -1))) / 2
        running_loss += loss.item()

        if plotPath is not None:

            x = np.linspace(140.25, 159.75, 40)
            y = np.linspace(25.25, 39.75, 30)
            if len(labels.size()) < 5:
                if 'transfer' in plotPath:
                    u = datasetClass.transfer_reduce(u.reshape(-1, 30, 40),'u')
                    v = datasetClass.transfer_reduce(v.reshape(-1, 30, 40),'v')
                    u_l = datasetClass.transfer_reduce(labels[:, 0, :],'u')
                    v_l = datasetClass.transfer_reduce(labels[:, 1, :],'v')
                    u_shape_collection = torch.concatenate((u_shape_collection,u.reshape(-1,1200)),dim=0)
                    v_shape_collection = torch.concatenate((v_shape_collection,v.reshape(-1,1200)),dim=0)
                    lu_shape_collection = torch.concatenate((lu_shape_collection,u_l.reshape(-1,1200)),dim=0)
                    lv_shape_collection = torch.concatenate((lv_shape_collection,v_l.reshape(-1,1200)),dim=0)
                else:
                    u = datasetClass.nc_reduce(u.reshape(-1, 30, 40),'u')
                    v = datasetClass.nc_reduce(v.reshape(-1, 30, 40),'v')
                    u_l = datasetClass.nc_reduce(labels[:, 0, :],'u')
                    v_l = datasetClass.nc_reduce(labels[:, 1, :],'v')
                    u_shape_collection = torch.concatenate((u_shape_collection,u.reshape(-1,1200)),dim=0)
                    v_shape_collection = torch.concatenate((v_shape_collection,v.reshape(-1,1200)),dim=0)
                    lu_shape_collection = torch.concatenate((lu_shape_collection,u_l.reshape(-1,1200)),dim=0)
                    lv_shape_collection = torch.concatenate((lv_shape_collection,v_l.reshape(-1,1200)),dim=0)
            else:
                if 'transfer' in plotPath:
                    u = datasetClass.transfer_reduce(u.reshape(-1, 30, 40),'u')
                    v = datasetClass.transfer_reduce(v.reshape(-1, 30, 40),'v')
                    u_l = datasetClass.transfer_reduce(labels[:,0, 0, :],'u')
                    v_l = datasetClass.transfer_reduce(labels[:,0, 1, :],'v')
                    u_shape_collection = torch.concatenate((u_shape_collection,u.reshape(-1,1200)),dim=0)
                    v_shape_collection = torch.concatenate((v_shape_collection,v.reshape(-1,1200)),dim=0)
                    lu_shape_collection = torch.concatenate((lu_shape_collection,u_l.reshape(-1,1200)),dim=0)
                    lv_shape_collection = torch.concatenate((lv_shape_collection,v_l.reshape(-1,1200)),dim=0)
                else:
                    u = datasetClass.nc_reduce(u.reshape(-1, 30, 40),'u')
                    v = datasetClass.nc_reduce(v.reshape(-1, 30, 40),'v')
                    u_l = datasetClass.nc_reduce(labels[:,0, 0, :],'u')
                    v_l = datasetClass.nc_reduce(labels[:,0, 1, :],'v')
                    u_shape_collection = torch.concatenate((u_shape_collection,u.reshape(-1,1200)),dim=0)
                    v_shape_collection = torch.concatenate((v_shape_collection,v.reshape(-1,1200)),dim=0)
                    lu_shape_collection = torch.concatenate((lu_shape_collection,u_l.reshape(-1,1200)),dim=0)
                    lv_shape_collection = torch.concatenate((lv_shape_collection,v_l.reshape(-1,1200)),dim=0)
            # 方差最大
            # u_location = [19, 9]
            # v_location = [19, 9]
            # 涡旋区1
            # u_location = [20, 20]
            # v_location = [20, 20]
            # 涡旋区2
            u_location = [18, 31]
            v_location = [18, 31]
            u_collection = u_collection + [item for item in u.to('cpu').numpy()[:, u_location[0], u_location[1]]]
            v_collection = v_collection + [item for item in v.to('cpu').numpy()[:, v_location[0], v_location[1]]]
            lu_collection = lu_collection + [item for item in u_l.to('cpu').numpy()[:, u_location[0], u_location[1]]]
            lv_collection = lv_collection + [item for item in v_l.to('cpu').numpy()[:, v_location[0], v_location[1]]]

            X, Y = np.meshgrid(x, y)
            for index in range(len(u)):
                plt.figure(dpi=200)
                plt.quiver(X, Y, np.ma.array(u[index, 0::1, 0::1].to('cpu').numpy(), mask=datasetClass.mask_array), np.ma.array(v[index, 0::1, 0::1].to('cpu').numpy(), mask=datasetClass.mask_array), units='width', color='red')
                plt.quiver(X, Y, np.ma.array(u_l[index, 0::1, 0::1].to('cpu').numpy(), mask=datasetClass.mask_array), np.ma.array(v_l[index, 0::1, 0::1].to('cpu').numpy(), mask=datasetClass.mask_array),units='width', color='blue')
                plt.xlabel('longitude(°)')
                plt.ylabel('latitude(°)')
                plt.savefig(f'{plotPath}/{1980 + math.floor((259 + 86 + pic_index) / 12)}年{(259 + 86 + pic_index) % 12 + 1}月海流.tif')
                plt.close()
                pic_index = pic_index+1
    if plotPath is not None:
        u_shape_collection = u_shape_collection.transpose(1, 0)
        lu_shape_collection = lu_shape_collection.transpose(1, 0)
        v_shape_collection = v_shape_collection.transpose(1, 0)
        lv_shape_collection = lv_shape_collection.transpose(1, 0)
        u_coff = []
        v_coff = []
        for index in range(len(u_shape_collection)):
            u_coff.append(torch.min(torch.corrcoef(
                torch.concatenate((u_shape_collection[index:index + 1, :], lu_shape_collection[index:index + 1, :]),
                                  dim=0))))
            v_coff.append(torch.min(torch.corrcoef(
                torch.concatenate((v_shape_collection[index:index + 1, :], lv_shape_collection[index:index + 1, :]),
                                  dim=0))))
        u_coff = np.array(torch.tensor(u_coff).reshape(30, 40))
        v_coff = np.array(torch.tensor(v_coff).reshape(30, 40))
    return running_loss / (i + 1), u_collection, lu_collection, v_collection, lv_collection
import matplotlib.pyplot as plt
import pandas as pd

# 保存计算的过程loss，并根据需要绘制loss曲线
class data_analyse:
    def __init__(self):
        self.test_result = []
        self.parameters = {}
        self.plotPath = "./result"
        self.model_name = None
        self.local_loss = []
        self.transfer_loss = []


    def refreshData(self, DMSE, MSE, transfer_MSE, new_parameters):
        for key in new_parameters.keys():
            if key in self.parameters.keys():
                self.parameters[key].append(new_parameters[key])
            else:
                self.parameters.update({key: [new_parameters[key]]})

        loss = {'DMSE':DMSE, 'MSE':MSE, 'transfer_MSE':transfer_MSE}
        for key in loss.keys():
            if key in self.parameters.keys():
                self.parameters[key].append(loss[key])
            else:
                self.parameters.update({key: [loss[key]]})
        self.model_name = new_parameters['model_name']

    def refreshCollections(self, u_collection, lu_collection, v_collection, lv_collection):
        self.parameters.append(new_parameters)
        self.test_result.append(new_test)
        self.model_name = new_parameters['model_name']

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(range(0, self.epochIndex + 1), self.lossData)
        plt.savefig(f'{self.plotPath}/损失函数图.png')
        print("绘图完毕")

    def output_parameters(self):
        df = pd.DataFrame(self.parameters)
        df.to_csv(f'{self.plotPath}/{self.model_name}.csv')

    def output_collections(self):
        df = pd.DataFrame({'parameters': self.parameters, 'test': self.test_result})
        df.to_csv(f'{self.plotPath}/{self.model_name}.csv')

    def output_to_excel(self, data_collections, fileName:str):

        df = pd.DataFrame({'u': data_collections[0],'lu': data_collections[1],'v': data_collections[2],'lv': data_collections[3]})
        df.to_csv(f'./result/{fileName}_out_{self.model_name}.csv')

    def refresh_loss(self, local_loss, transfer_loss):
        self.local_loss.append(local_loss)
        self.transfer_loss.append(transfer_loss)


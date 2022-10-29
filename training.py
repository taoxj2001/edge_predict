import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import Normalizer
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from dataset_loader import DataLoader
from models import *
from predict_edge import Net
import torch
from math import log
from torch_geometric.datasets import WikipediaNetwork
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from decimal import *
from GMM import GMM
if __name__ == '__main__':
    '''载入data'''
    # dataset = WikipediaNetwork(root = "/data/squirrel", name = "squirrel")
    # dataset = WikipediaNetwork(root="/data/chameleon", name="chameleon")
    dataset = DataLoader("texas")
    data = dataset[0]
    device = torch.device('cpu')

    def floatrange(start, stop, steps):
        '''
        start:计数从 start 开始
        stop:计数到 stop 结束
        step:步长
        '''
        resultList = []
        while Decimal(str(start)) <= Decimal(str(stop)):
            resultList.append(float(Decimal(str(start))))
            start = Decimal(str(start)) + Decimal(str(steps))
        return resultList
    # 按照固定区间长度绘制频率分布直方图
    # bins_interval 区间的长度
    # margin    设定的左边和右边空留的大小
    def probability_distribution(data,data2):
        mindata1 = min(data)
        maxdata1 = max(data)
        mindata2 = min(data2)
        maxdata2 = max(data2)
        mindata = min(mindata1, mindata2)
        maxdata = max(maxdata1, maxdata2)
        bins_interval = (maxdata - mindata)/2000
        bins = floatrange(mindata, maxdata + bins_interval, bins_interval)
        plt.xlim(mindata, maxdata)
        plt.title("probability-distribution")
        plt.xlabel('Interval')
        plt.ylabel('Probability')
        plt.hist(x=data, bins=bins, histtype='stepfilled', color='r', label= "homophilic", alpha= 0.3)
        plt.hist(x=data2, bins=bins, histtype='stepfilled', color='b', label="heterophilic", alpha= 0.3)
        plt.show()

    def celoss1(score, label):

        if label == 1:
            label1 = [1, 0]
        else:
            label1 = [0, 1]
        temp = [-log(score), -log(1 - score)]
        celoss = label1[0] * temp[0] + label1[1] * temp[1]
        return celoss


    def train():
        model.train()
        optimizer.zero_grad()
        loss_list = []
        z = model.encode(train_data.x, train_data.edge_index)
        edge_label_index = train_data.edge_label_index
        edge_label = train_data.edge_label
        out = model.decode(z, edge_label_index).view(-1)
        out_sigmoid = out.sigmoid()
        for i in range(int(len(out)/2)):
            if out_sigmoid[i] == 1:
                out_sigmoid[i] = 9.999e-01
            elif out_sigmoid[i] == 0:
                out_sigmoid[i] = 1e-03
            loss_list.append(celoss1(out_sigmoid[i], edge_label[i]))
        loss_list = np.array(loss_list)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        return loss, loss_list


    @torch.no_grad()
    def test(data):
        model.eval()
        temp = model.encode(data.x, data.edge_index)
        out = model.decode(temp, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    runs = 1
    results = []
    for run in range(runs):
        model = Net(data.num_features, 256, 128).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(num_val=0.05, num_test=0.05, is_undirected=True,
                              add_negative_train_samples=True, neg_sampling_ratio=1.0),
        ])
        train_data, val_data, test_data = transform(data)
        best_val_acc = final_test_acc = 0
        len_data = train_data.edge_label_index.size()[1]
        loss_sum_list = []
        for i in range(int(len_data/2)):
            loss_sum_list.append(0)
        loss_list = []
        max_epoch = 10
        for epoch in range(max_epoch):
            loss, loss_list_i = train()
            loss_list.append(loss_list_i)
            val_acc = test(val_data)
            test_acc = test(test_data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
            for i in range(int(len_data/2)):
                loss_sum_list[i] += loss_list_i[i]
        print(loss_sum_list)
        #print(loss_list)


        z = model.encode(train_data.x, train_data.edge_index)
        edge_label_index = train_data.edge_label_index
        edge_label = train_data.edge_label
        # print(edge_label)
        out = model.decode(z, edge_label_index).view(-1)
        out_sigmoid = out.sigmoid()
        # print(out)
        # print(len(out_sigmoid))



        loss2 = list(range(int(len(out)/2)))
        for i in range(int(len(out)/2)):
            if out_sigmoid[i] == 1:
                out_sigmoid[i] = 9.999e-01
            elif out_sigmoid[i] == 0:
                out_sigmoid[i] = 1e-03
            loss2[i] = celoss1(out_sigmoid[i], edge_label[i])
        # print(loss2)

        # loss2 = loss_sum_list
        loss2 = np.array(loss2)

        # print(type(loss3))
        loss3 = loss2.reshape(-1, 1)
        # loss3 = Normalizer().fit_transform(loss3)
        gmm = GMM(loss3, 2)
        print(loss3)
        gmm.GMM_EM()
        y_pre = gmm.prediction
        len_pre = len(y_pre)
        homo = []
        hetero = []
        for i in range(len_pre):
            if y_pre[i] == 0:
                homo.append(loss2[i])
            else:
                hetero.append(loss2[i])
        probability_distribution(homo, hetero)
        print(f'Final Test: {final_test_acc:.4f}')
        results.append(final_test_acc)
    results = 100 * torch.Tensor(results)
    print(results)
    print(f'Averaged test accuracy for {runs} runs: {results.mean():.2f} \pm {results.std():.2f}')

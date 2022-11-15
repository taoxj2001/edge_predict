import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import Normalizer
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling, add_self_loops
import torch_geometric.transforms as T
from dataset_loader import DataLoader
from models import *
from predict_edge import Net
import torch
from math import log
from torch_geometric.datasets import WikipediaNetwork, Actor, Coauthor
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from decimal import *
from GMM import GMM
import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits
from models import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time


def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        reg_loss = None
        loss.backward()
        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    tmp_net = Net(dataset, args)

    # randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)

    model, data = tmp_net.to(device), data.to(device)

    if args.net == 'GPRGNN':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net == 'BernNet':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        train(model, optimizer, data, args.dprate)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:', epoch)
                    break
    return test_acc, best_val_acc, theta, time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')

    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR',
                        help='initialization for GPRGNN.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')


    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')

    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
    parser.add_argument('--max_epoch', type=int, default=35, help='max epoch for edge classification')
    parser.add_argument('--delete_ratio', type=float, default=0.1)
    parser.add_argument('--neg_sampling_ratio', type=float, default=3.0)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'GPRGNN', 'BernNet', 'MLP'],
                        default='GCN')
    args = parser.parse_args()
    '''载入data'''
    # dataset = WikipediaNetwork(root="/data/squirrel", name="squirrel")
    # dataset = WikipediaNetwork(root="/data/chameleon", name="chameleon")
    # dataset = Actor(root="/data/actor")
    # dataset = DataLoader("cornell")
    # dataset = Coauthor(root="/data/physics", name='Physics')
    dataset = DataLoader("cora")
    data = dataset[0]
    device = torch.device('cpu')
    data.edge_index, data.edge_weight = add_self_loops(data.edge_index)



    def node_class_query(i, j, dataset):
        iclass = dataset.data.y[i]
        jclass = dataset.data.y[j]
        if iclass == jclass:
            return 0  # homophilic
        else:
            return 1  # heterophilic

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
    def probability_distribution(data, data2=None):
        step = 80
        if data2 == None:
            mindata = min(data)
            maxdata = max(data)
            bins_interval = (maxdata - mindata) / step
            bins = floatrange(mindata, maxdata + bins_interval, bins_interval)
            plt.xlim(mindata, maxdata)
            plt.title("probability-distribution1")
            plt.xlabel('Interval')
            plt.ylabel('Probability')
            plt.hist(x=data, bins=bins, histtype='stepfilled', color='r', label="homophilic", alpha=0.3, density=False)
            plt.show()
        else:
            mindata1 = min(data)
            maxdata1 = max(data)
            mindata2 = min(data2)
            maxdata2 = max(data2)
            mindata = min(mindata1, mindata2)
            maxdata = max(maxdata1, maxdata2)
            bins_interval = (maxdata - mindata) / step
            bins = floatrange(mindata, maxdata + bins_interval, bins_interval)
            plt.xlim(mindata, maxdata)
            plt.title("probability-distribution2")
            plt.xlabel('Interval')
            plt.ylabel('Probability')
            plt.hist(x=data, bins=bins, histtype='stepfilled', color='r', label="homophilic", alpha=0.3, density=False)
            plt.hist(x=data2, bins=bins, histtype='stepfilled', color='b', label="heterophilic", alpha=0.3,
                     density=False)
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
        z = model.encode(train_data.x, train_data.edge_label_index)
        edge_label_index = train_data.edge_label_index
        edge_label = train_data.edge_label
        out = model.decode(z, edge_label_index).view(-1)
        out_sigmoid = out.sigmoid()
        penalty = 0

        for i in range(len(out)):
            penalty += out_sigmoid[i] * log(out_sigmoid[i])
        out_sigmoid = out_sigmoid.detach()
        edge_label.detach()
        for i in range(pos_len):
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
        temp = model.encode(data.x, data.edge_label_index)
        out = model.decode(temp, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


    runs = 1
    results = []
    for run in range(runs):
        model = Net(data.num_features, 256, 64).to(device)
        max_epoch = args.max_epoch
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(num_val=0.05, num_test=0.05, is_undirected=True,
                              add_negative_train_samples=True, neg_sampling_ratio=args.neg_sampling_ratio),
        ])
        train_data, val_data, test_data = transform(data)
        pos_len = 0
        for i in range(int(train_data.edge_label.shape[0])):
            if int(train_data.edge_label[i]) == 1:
                pos_len += 1

        best_val_acc = final_test_acc = 0
        len_data = train_data.edge_label_index.size()[1]
        loss_sum_list = []
        for i in range(pos_len):
            loss_sum_list.append(0)
        loss_list = []
        # homoedge = 0
        # heteroedge = 0
        # for i in range(int(len_data/2)):
        #     a = train_data.edge_label_index[0][i]
        #     b = train_data.edge_label_index[1][i]
        #     if node_class_query(a, b, dataset) == 1:
        #         homoedge += 1
        #     else:
        #         heteroedge += 1
        # print("homo = ", homoedge)
        # print("hetero = ", heteroedge)

        for epoch in range(max_epoch):
            loss, loss_list_i = train()
            loss_list.append(loss_list_i)
            val_acc = test(val_data)
            test_acc = test(test_data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
            for i in range(pos_len):
                loss_sum_list[i] += loss_list_i[i]
        # print(loss_sum_list)
        loss_list_r = []
        for i in range(pos_len):
            list_temp = np.arange(max_epoch) * 1.0
            for epoch in range(max_epoch):
                list_temp[epoch] = loss_list[epoch][i]
            loss_list_r.append(list_temp)
        '''计算每一条边贡献的交叉熵损失'''
        model.eval()
        z = model.encode(train_data.x, train_data.edge_label_index)
        edge_label_index = train_data.edge_label_index
        edge_label = train_data.edge_label
        out = model.decode(z, edge_label_index).view(-1)
        out_sigmoid = out.sigmoid()
        out_sigmoid = out_sigmoid.detach()
        loss2 = list(range(pos_len))
        for i in range(pos_len):
            if out_sigmoid[i] == 1:
                out_sigmoid[i] = 9.999e-01
            elif out_sigmoid[i] == 0:
                out_sigmoid[i] = 1e-03
            loss2[i] = celoss1(out_sigmoid[i], edge_label[i])

        loss2 = np.array(loss2)
        loss3 = loss2.reshape(-1, 1)

        '''将同嗜边和异嗜边对应的交叉熵损失存在两个list中'''
        homo = []
        hetero = []
        for i in range(pos_len):
            a = train_data.edge_label_index[0][i]
            b = train_data.edge_label_index[1][i]
            if a != b:
                if node_class_query(a, b, dataset) == 0:
                    homo.append(loss2[i])
                else:
                    hetero.append(loss2[i])
        '''画图'''
        # probability_distribution(homo, hetero)
        # probability_distribution(loss2)
        delete_ratio = args.delete_ratio
        delete_num = int(delete_ratio * pos_len)
        temploss = list(loss2)
        delete_list = []

        for i in range(delete_num):
            k = temploss.index(max(temploss))
            delete_list.append(k)
            temploss[k] = 0
        delete_edge = torch.zeros(2, len(delete_list), )

        for i in range(len(delete_list)):
            n = delete_list[i]
            delete_edge[0][i] = int(train_data.edge_label_index[0][n])
            delete_edge[1][i] = int(train_data.edge_label_index[1][n])
        edge_new_index = torch.tensor([], dtype=torch.int64)
        for edge in range(data.num_edges):
            flag = 0
            temp = torch.tensor([0])
            for i in range(len(delete_list)):
                if data.edge_index[0][edge] == delete_edge[0][i] and data.edge_index[1][edge] == delete_edge[1][i]:
                    flag = 1
                if data.edge_index[1][edge] == delete_edge[0][i] and data.edge_index[0][edge] == delete_edge[1][i]:
                    flag = 1
            if flag == 0:  # not in delete list
                temp = torch.tensor([edge], dtype=torch.int64)
                edge_new_index = torch.cat([edge_new_index, temp], dim=-1)
        new_edge = torch.index_select(data.edge_index, 1, edge_new_index)

        # '''使用GMM模型进行分类'''
        # gmm = GMM(loss3, 2)
        # print(loss3)
        # gmm.GMM_EM()
        # y_pre = gmm.prediction
        # print(y_pre)
        # len_pre = len(y_pre)
        # homo2 = []
        # hetero2 = []
        # for i in range(len_pre):
        #     if y_pre[i] == 0:
        #         homo2.append(loss2[i])
        #     else:
        #         hetero2.append(loss2[i])
        # probability_distribution(homo2, hetero2)

        # '''损失变化折线图'''
        # y = np.arange(max_epoch)
        # for i in range(int(len(out)/2)):
        #     a = train_data.edge_label_index[0][i]
        #     b = train_data.edge_label_index[1][i]
        #     if node_class_query(a,b,dataset) == 0:
        #         plt.plot(y,loss_list_r[i], color='r', linewidth=0.1)
        #     else:
        #         plt.plot(y,loss_list_r[i], color='b', linewidth=0.1)
        # plt.show()

        # for i in range(len_pre):
        #     if y_pre[i] == 0:
        #         plt.plot(y,loss_list_r[i], color='r', linewidth=0.1)
        #     else:
        #         plt.plot(y,loss_list_r[i], color='b', linewidth=0.1)
        # plt.show()

        # list1 = []
        # list0 = []
        # for i in range(len_pre):
        #     if y_pre[i] == 0:
        #         for epoch in range(max_epoch):
        #             list0.append(loss_list[epoch][i])
        #     else:
        #         for epoch in range(max_epoch):
        #             list1.append(loss_list[epoch][i])
        print(f'Final Test: {final_test_acc:.4f}')
        results.append(final_test_acc)

        '''node classification'''

        # 10 fixed seeds for splits


        args.weight_decay = 0.0005
        args.lr = 0.05  # learning rate
        args.dropout = 0.5  # dropout for neural networks
        args.hidden = 64
        args.K = 10
        args.Bern_lr = 0.01  # learning rate for BernNet propagation layer
        args.dprate = 0.0  # dropout for propagation layer

        print(args)
        print("---------------------------------------------")
        gnn_name = args.net
        if gnn_name == 'GCN':
            Net = GCN_Net
        elif gnn_name == 'GAT':
            Net = GAT_Net
        elif gnn_name == 'APPNP':
            Net = APPNP_Net
        elif gnn_name == 'ChebNet':
            Net = ChebNet
        elif gnn_name == 'GPRGNN':
            Net = GPRGNN
        elif gnn_name == 'BernNet':
            Net = BernNet
        elif gnn_name == 'MLP':
            Net = MLP

        print(data)
        '''old edge'''
        percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
        val_lb = int(round(args.val_rate * len(data.y)))
        SEEDS = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661, 1648766618, 629014539,
                 3212139042, 1424918363, 41488137, 4100006517, 983997847, 4023022221, 2019585660, 2108550661, 1648766618, 621014539,
                 3000039042, 2111118363, 1941588137, 4196517, 981145847, 4023022221, 4019585660, 2108550661, 1648000618, 629010539,
                 3212139042, 2424018363, 1941111137, 4198936517, 985111847, 4023022221, 4010005660, 2108550661, 1648766618, 629014539,
                 3200001042, 2424818363, 1941488137, 4198150087, 983979847, 4023022001, 4010835660, 2108550661, 1648766618, 629014039,
                 3212139042, 2424918363]
        results = []
        time_results = []
        for RP in tqdm(range(args.runs)):
            args.seed = SEEDS[RP]
            test_acc, best_val_acc, theta_0, time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)
            time_results.append(time_run)
            results.append([test_acc, best_val_acc, theta_0])
            print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')
            if args.net == 'BernNet':
                print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])

        run_sum = 0
        epochsss = 0
        for i in time_results:
            run_sum += sum(i)
            epochsss += len(i)

        print("each run avg_time:", run_sum / (args.runs), "s")
        print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

        test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

        values = np.asarray(results)[:, 0]
        uncertainty = np.max(
            np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
        # print(uncertainty*100)
        print(f'{gnn_name} on dataset {dataset.name}, in {args.runs} repeated experiment:')
        print(" old edge ")
        print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')


        '''new edge'''
        data.edge_index = new_edge
        print(data)
        percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
        val_lb = int(round(args.val_rate * len(data.y)))

        results = []
        time_results = []
        for RP in tqdm(range(args.runs)):
            args.seed = SEEDS[RP]
            test_acc, best_val_acc, theta_0, time_run = RunExp(args, dataset, data, Net, percls_trn, val_lb)
            time_results.append(time_run)
            results.append([test_acc, best_val_acc, theta_0])
            print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')
            if args.net == 'BernNet':
                print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])

        run_sum = 0
        epochsss = 0
        for i in time_results:
            run_sum += sum(i)
            epochsss += len(i)

        print("each run avg_time:", run_sum / (args.runs), "s")
        print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

        test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

        values = np.asarray(results)[:, 0]
        uncertainty = np.max(
            np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))
        # print(uncertainty*100)
        print(f'{gnn_name} on dataset {dataset.name}, in {args.runs} repeated experiment:')
        print("new edge")
        print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

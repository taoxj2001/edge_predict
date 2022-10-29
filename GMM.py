import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

'''only for one-dimension feature'''

class GMM:
    def __init__(self, Data, K, weights=None, means=None, vars=None):
        """
        这是GMM（高斯混合模型）类的构造函数
        :param Data: 训练数据
        :param K: 高斯分布的个数
        :param weigths: 每个高斯分布的初始概率（权重）
        :param means: 高斯分布的均值向量
        :param vars: 高斯分布的方差集合
        """
        self.Data = Data
        self.K = K
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)  # 归一化
        if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K):
                mean = np.random.rand()*10
                self.means.append(mean)
            print(self.means)

        if vars is not None:
            self.vars = vars
        else:
            self.vars = []
            for i in range(self.K):
                var = np.random.rand()*10
                self.vars.append(var)

    def Gaussian(self, x, mean, var):
        if var == 0:
            var = var + 0.001
        xdiff = (x - mean)
        prob = 1.0 / np.power(2 * np.pi * np.abs(var) * np.abs(var), 0.5) * np.exp(-0.5 * xdiff*(var**(-1))*(xdiff))
        return prob

    def GMM_EM(self):
        """
        这是利用EM算法进行优化GMM参数的函数
        :return: 返回各组数据的属于每个分类的概率
        """
        loglikelyhood = 0
        oldloglikelyhood = 1
        len = np.shape(self.Data)[0]
        dim = 1
        # gamma表示第n个样本属于第k个混合高斯的概率
        gammas = [np.zeros(self.K) for i in range(len)]
        while np.abs(loglikelyhood - oldloglikelyhood) > 0.001:
            print(np.abs(loglikelyhood - oldloglikelyhood))
            oldloglikelyhood = loglikelyhood
            # E-step
            for n in range(len):
                # respons是GMM的EM算法中的权重w，即后验概率
                respons = [self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.vars[k])
                           for k in range(self.K)]
                respons = np.array(respons)
                sum_respons = np.sum(respons)
                gammas[n] = respons / sum_respons
            # M-step
            for k in range(self.K):
                # nk表示N个样本中有多少属于第k个高斯
                nk = np.sum([gammas[n][k] for n in range(len)])
                # 更新每个高斯分布的概率
                self.weights[k] = 1.0 * nk / len
                # 更新高斯分布的均值
                self.means[k] = (1.0 / nk) * np.sum([gammas[n][k] * self.Data[n] for n in range(len)])
                xdiffs = self.Data - self.means[k]
                # 更新高斯分布的方差
                self.vars[k] = (1.0 / nk) * np.sum([gammas[n][k] * xdiffs[n] * xdiffs[n] for n in range(len)])

            loglikelyhood1 = []

            for n in range(len):
                tmp = [np.sum(self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.vars[k])) for k in range(self.K)]
                tmp = np.log(np.array(tmp))
                loglikelyhood1.append(list(tmp))
            loglikelyhood = np.sum(loglikelyhood1) / len

        for i in range(len):
            gammas[i] = gammas[i] / np.sum(gammas[i])

        self.posibility = gammas
        self.prediction = [np.argmax(gammas[i]) for i in range(len)]


# def run_main():
#     """
#         这是主函数
#     """
#     # 导入Iris数据集
#     iris = load_iris()
#     label = np.array(iris.target)
#     data = np.array(iris.data)
#     print("Iris数据集的标签：\n", label)
#
#     # 对数据进行预处理
#     data = Normalizer().fit_transform(data)
#
#     print(type(data))
#
#     # 解决画图是的中文乱码问题
#     mpl.rcParams['font.sans-serif'] = [u'simHei']
#     mpl.rcParams['axes.unicode_minus'] = False
#
#     # 数据可视化
#     plt.scatter(data[:, 0], data[:, 1], c=label)
#     plt.title("Iris数据集显示")
#     plt.show()
#
#     # GMM模型
#     K = 3
#     gmm = GMM(data, K)
#     gmm.GMM_EM()
#     y_pre = gmm.prediction
#     print("GMM预测结果：\n", y_pre)
#     print("GMM正确率为：\n", accuracy_score(label, y_pre))
#     plt.scatter(data[:, 0], data[:, 1], c=y_pre)
#     plt.title("GMM结果显示")
#     plt.show()
# # run_main()
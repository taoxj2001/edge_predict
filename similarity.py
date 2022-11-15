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
import torch.nn.functional as F




dataset = WikipediaNetwork(root = "/data/squirrel", name = "squirrel")
# dataset = WikipediaNetwork(root="/data/chameleon", name="chameleon")
dataset = Actor(root="/data/actor")
# dataset = DataLoader("texas")
# dataset = Coauthor(root="/data/physics", name='Physics')
dataset = DataLoader("cora")
data = dataset[0]
data.edge_weight = torch.ones(int(data.edge_index.shape[1]))
device = torch.device('cpu')
data.edge_index, data.edge_weight = add_self_loops(data.edge_index)
print(data)
print(data.y[3])
for i in range(100):
    print(F.cosine_similarity(data.x[3], data.x[i], dim=0), data.y[i])

import scanpy as sc
import numpy as np
import torch
import copy
import random
from torch.utils.data import Dataset
import pandas as pd
import os
# from torch_geometric.data import Data
# from math import sqrt
# from torch_geometric.nn import GCNConv


class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y

    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx],
                'label': self.train_y[idx]}


def load_data(dir):
    if os.path.exists("preprocess/data/" + dir + "_preprocessed_counts.csv"):
        data_filter = pd.read_csv("preprocess/data/" + dir + "_preprocessed_counts.csv", header=0).iloc[:, 1:].to_numpy()
        label_filter = pd.read_csv("preprocess/data/" + dir + "_preprocessed_labels.csv", header=0).iloc[:, 1].tolist()
        # label_dirc = {k: v for k, v in zip(label_filter, range(len(label_filter)))}
        label_dirc = {}
        label_dirc_num = 0
        for k in label_filter:
            if k in label_dirc:
                continue
            else:
                label_dirc[k] = label_dirc_num
                label_dirc_num += 1

        label_save = []
        for id in label_filter:
            for k, v in label_dirc.items():
                if id == k:
                    label_save.append(v)

        label_filter = np.array(label_save)
        print("label_dirc:", label_dirc)
        print("label_dirc length:", label_dirc.__len__())


    else:
        adata = sc.read_h5ad("preprocess/data/" + dir + ".h5ad")
        print(dir, ": ", adata)

        # get X_dataset
        print("adata.X", adata.X)
        dataset = adata.X.toarray()
        print("got X, the shape is ", dataset.shape)

        # get label
        label = adata.obs["cell_ontology_class"]
        unique_label = list(set(label))
        label_dirc = {k: v for k, v in zip(unique_label, range(len(unique_label)))}
        print("label_dirc:", label_dirc)

        highly_variable = adata.var['highly_variable']._values

        label_index = []
        for k, v in label.items():
            label_index.append(label_dirc[v])

        label_index = np.array(label_index)

        # filter out low-quality cells with fewer than 200 genes and genes expressed in less than 3 cells
        data_filter, label_filter = filter_data(dataset, label_index, highly_variable)

        label_save = []
        for id in label_filter:
            for k, v in label_dirc.items():
                if id == v:
                    label_save.append(str(k))
        label_save = np.expand_dims(np.array(label_save), axis=1)
        dataframe = pd.DataFrame(np.concatenate((label_save, data_filter), axis=1))
        dataframe.to_csv("preprocess/data/" + dir + "_preprocessed_counts.csv", index=False, sep=',')

        dataframe = pd.DataFrame(label_save)
        dataframe.to_csv("preprocess/data/" + dir + "_preprocessed_labels.csv", index=True, sep=',')





    key = np.unique(label_filter)
    results = {}
    max_num = 0
    min_num = 100000
    for k in key:
        v = label_filter[label_filter == k].size
        results[k] = v
        if max_num < v:
            max_num = v
        if min_num > v:
            min_num = v
    #
    # print("The counts of each class after filter:", results)
    # print("samples num:", data_filter.shape[0])
    # print("gene num:", data_filter.shape[1])
    # print("Max_num:", max_num)
    # print("Min_num:", min_num)

    # preprocessed the X_dataset.. augmentation
    aug1 = []
    aug2 = []
    gene_num = data_filter.shape[1]

    for x in data_filter:
        aug1.append(random_gaussian_noise(x, gene_num))
        aug2.append(random_gaussian_noise(x, gene_num))

    train_data1 = torch.tensor(torch.from_numpy(data_filter).unsqueeze(1), dtype=torch.float32)
    train_data2 = torch.tensor(torch.from_numpy(np.array(aug1)).unsqueeze(1), dtype=torch.float32)
    train_data3 = torch.tensor(torch.from_numpy(np.array(aug2)).unsqueeze(1), dtype=torch.float32)

    train_lable = torch.from_numpy(label_filter)
    torch_dataset = ExplitAugSamples(train_data1, train_data2, train_data3, train_lable)

    return torch_dataset, gene_num, len(results), label_dirc


# def load_geometric_data(dir):
#     if os.path.exists("preprocess/data/" + dir + "_preprocessed_counts.csv"):
#         data_filter = pd.read_csv("preprocess/data/" + dir + "_preprocessed_counts.csv", header=0).iloc[:, 1:].to_numpy()
#         label_filter = pd.read_csv("preprocess/data/" + dir + "_preprocessed_labels.csv", header=0).iloc[:, 1].tolist()
#         # label_dirc = {k: v for k, v in zip(label_filter, range(len(label_filter)))}
#         label_dirc = {}
#         label_dirc_num = 0
#         for k in label_filter:
#             if k in label_dirc:
#                 continue
#             else:
#                 label_dirc[k] = label_dirc_num
#                 label_dirc_num += 1
#
#         label_save = []
#         for id in label_filter:
#             for k, v in label_dirc.items():
#                 if id == k:
#                     label_save.append(v)
#
#         label_filter = np.array(label_save)
#         print("label_dirc:", label_dirc)
#         print("label_dirc length:", label_dirc.__len__())
#
#
#     else:
#         adata = sc.read_h5ad("preprocess/data/" + dir + ".h5ad")
#         print(dir, ": ", adata)
#
#         # get X_dataset
#         print("adata.X", adata.X)
#         dataset = adata.X.toarray()
#         print("got X, the shape is ", dataset.shape)
#
#         # get label
#         label = adata.obs["cell_ontology_class"]
#         unique_label = list(set(label))
#         label_dirc = {k: v for k, v in zip(unique_label, range(len(unique_label)))}
#         print("label_dirc:", label_dirc)
#
#         highly_variable = adata.var['highly_variable']._values
#
#         label_index = []
#         for k, v in label.items():
#             label_index.append(label_dirc[v])
#
#         label_index = np.array(label_index)
#
#         # filter out low-quality cells with fewer than 200 genes and genes expressed in less than 3 cells
#         data_filter, label_filter = filter_data(dataset, label_index, highly_variable)
#
#         label_save = []
#         for id in label_filter:
#             for k, v in label_dirc.items():
#                 if id == v:
#                     label_save.append(str(k))
#         label_save = np.expand_dims(np.array(label_save), axis=1)
#         dataframe = pd.DataFrame(np.concatenate((label_save, data_filter), axis=1))
#         dataframe.to_csv("preprocess/data/" + dir + "_preprocessed_counts.csv", index=False, sep=',')
#
#         dataframe = pd.DataFrame(label_save)
#         dataframe.to_csv("preprocess/data/" + dir + "_preprocessed_labels.csv", index=True, sep=',')
#     key = np.unique(label_filter)
#     results = {}
#     max_num = 0
#     min_num = 100000
#     for k in key:
#         v = label_filter[label_filter == k].size
#         results[k] = v
#         if max_num < v:
#             max_num = v
#         if min_num > v:
#             min_num = v
#
#     print("The counts of each class after filter:", results)
#     print("samples num:", data_filter.shape[0])
#     print("gene num:", data_filter.shape[1])
#     print("Max_num:", max_num)
#     print("Min_num:", min_num)
#
#     # preprocessed the X_dataset as geometric data
#
#     X = torch.tensor(torch.from_numpy(data_filter), dtype=torch.float32)
#
#     # 每个结点的标签， 假设我们只有两个标签0和1
#     Y = torch.from_numpy(label_filter).unsqueeze(1)
#
#     # 边
#
#     sample_num = data_filter.shape[0]
#     edge_index_1 = []
#     edge_index_2 = []
#       # 用来记录x到样本数据集中每个点的距离
#
#     for i in range(sample_num):
#         distances = []
#         position = []
#         for j in range(sample_num):
#             if i == j:
#                 distances.append(1000000)
#                 position.append(-1)
#                 continue
#             d = sqrt(np.sum((data_filter[i] - data_filter[j]) ** 2))
#             distances.append(d)
#             position.append(j)
#
#         idex = np.array(distances).argsort()
#         for n in range(15):
#             edge_index_1.append(i)
#             edge_index_2.append(idex[n])
#
#         # nearest = np.argsort(distances)
#         # topK_y = [i for i in nearest[: 10]]
#
#     edge_index = torch.tensor([edge_index_1,
#                                edge_index_1], dtype=torch.long)
#     # 创建图
#     data = Data(x=X, edge_index=edge_index, y=Y)
#     conv = GCNConv(data_filter.shape[1], 16).cuda()
#     x_feature = conv(data.x.cuda(), data.edge_index.cuda())
#     print(x_feature)
#     gene_num = data_filter.shape[1]
#
#
#     return data, gene_num, len(results), label_dirc


def random_gaussian_noise(cell_profile, gene_num,p=0.8):
    # create the noise
    new_cell_profile =copy.deepcopy(cell_profile)
    noise = np.random.normal(0, 0.5, int(gene_num))
    mask = random.sample(range(0, int(gene_num)), int(gene_num*p))
    for index, id in enumerate(mask):
        new_cell_profile[id] += noise[index]

    return new_cell_profile

def filter_data(dataset, label, highly_variable):

    dataset = dataset[:, highly_variable]
    new_cell = []
    new_label = []
    gene_num = dataset.shape[1]
    for index, data in enumerate(dataset):
        sum_gene = 0
        for i in range(gene_num):
            if data[i] > 0:
                sum_gene += 1
        if sum_gene > 200:
            new_cell.append(data)
            new_label.append(label[index])

    new_cell = np.array(new_cell).T
    new_cell_gene = []
    new_cell_num = new_cell.shape[1]
    for data_cell in new_cell:
        sum_cell = 0
        for j in range(new_cell_num):
            if data_cell[j] > 0:
                sum_cell += 1
        if sum_cell > 3:
            new_cell_gene.append(data_cell)

    return np.array(new_cell_gene).T, np.array(new_label)

if __name__ == '__main__':
    datasets = ['yan']
    for da in datasets:
        print(da)
        torch_dataset, dim, num_classes, label_dirc = load_data(da)
        print("sample numb:", len(torch_dataset))
        # print("gene numb:", dim)
        # print("\n")

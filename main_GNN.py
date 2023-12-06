
import os
import random
from learner.contrastive_utils import central_PairConLoss, PairConLoss
import create_img
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import sys
sys.path.append('./')
import torch
import argparse
from utils.Kmeans import get_kmeans_centers, get_confusion
from utils.logger import setup_path, set_global_random_seed
from sklearn.cluster import KMeans
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Parameter
import torch_geometric.nn
import pandas as pd
import numpy as np
import scanpy as sc
from math import sqrt
from torch_geometric.data import Data
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.dropout import Dropout
import torch.nn.functional as F
from utils.metric import Confusion


class encoder(nn.Module):
    def __init__(self, args, dim, out_dim, device="cuda"):
        super(encoder, self).__init__()
        self.device = device
        self.args = args

        self.dim = dim

        # embedding coder

        self.conv1 = torch_geometric.nn.GCNConv(dim, 1024)
        self.norm1 = LayerNorm(1024)
        self.dropout1 = Dropout(0.2)

        self.conv2 = torch_geometric.nn.GCNConv(1024, 512)
        self.norm2 = LayerNorm(512)
        self.dropout2 = Dropout(0.2)

        self.conv3 = torch_geometric.nn.GCNConv(512, out_dim)
        self.norm3 = LayerNorm(out_dim)
        self.dropout3 = Dropout(0.2)



    def forward(self, data_batch, edge):
        x1 = self.norm1(self.dropout1(F.relu(self.conv1(data_batch, edge))))
        x2 = self.norm2(self.dropout1(F.relu(self.conv2(x1, edge))))
        x3 = self.norm3(self.dropout1(F.relu(self.conv3(x2, edge))))
        return x3


class Model(nn.Module):
    def __init__(self, args, dim, cluster_centers=None, alpha=1.0, encoder=None, device="cuda"):
        super(Model, self).__init__()
        self.encoder = encoder
        self.device = device
        self.args = args

        self.alpha = alpha

        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 256)
        )

        # Clustering head

        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, data_batch):
        return self.contrast_head(data_batch)

    def get_cluster_prob(self, embeddings, cluster_centers):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - cluster_centers.to(self.device, non_blocking=True)) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


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


def random_gaussian_noise(cell_profile, gene_num):
    # create the noise
    noise = np.random.normal(0, 0.5, int(gene_num))
    mask = random.sample(range(0, int(gene_num)), int(gene_num))
    for index, id in enumerate(mask):
        cell_profile[id] += noise[index]

    return cell_profile


def get_edge(data_filter):
    sample_num = data_filter.shape[0]
    edge_index_1 = []
    edge_index_2 = []
    # 用来记录x到样本数据集中每个点的距离

    for i in range(sample_num):
        distances = []
        position = []
        for j in range(sample_num):
            if i == j:
                distances.append(1000000)
                position.append(-1)
                continue
            d = sqrt(np.sum((data_filter[i] - data_filter[j]) ** 2))
            distances.append(d)
            position.append(j)

        idex = np.array(distances).argsort()
        for n in range(5):
            edge_index_1.append(i)
            edge_index_2.append(idex[n])

        # nearest = np.argsort(distances)
        # topK_y = [i for i in nearest[: 10]]

    edge_index = torch.tensor([edge_index_1,
                               edge_index_1], dtype=torch.long)
    return edge_index

def load_geometric_data(dir):
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

    print("The counts of each class after filter:", results)
    print("samples num:", data_filter.shape[0])
    print("gene num:", data_filter.shape[1])
    print("Max_num:", max_num)
    print("Min_num:", min_num)

    # preprocessed the X_dataset as geometric data
    aug1 = []
    aug2 = []
    gene_num = data_filter.shape[1]
    for x in data_filter:
        aug1.append(random_gaussian_noise(x, gene_num))
        aug2.append(random_gaussian_noise(x, gene_num))

    X = torch.tensor(torch.from_numpy(data_filter), dtype=torch.float32)
    aug_X1 = torch.tensor(torch.from_numpy(np.array(aug1)), dtype=torch.float32)
    aug_X2 = torch.tensor(torch.from_numpy(np.array(aug2)), dtype=torch.float32)

    # 每个结点的标签， 假设我们只有两个标签0和1
    Y = torch.from_numpy(label_filter).unsqueeze(1)

    # 边
    edge_index = get_edge(data_filter)
    # 创建图
    data = Data(x=X, edge_index=edge_index, y=Y)

    # 边
    edge_index1 = get_edge(np.array(aug1))
    # 创建图
    data1 = Data(x=aug_X1, edge_index=edge_index1, y=Y)

    # 边
    edge_index2 = get_edge(np.array(aug2))
    # 创建图
    data2 = Data(x=aug_X2, edge_index=edge_index2, y=Y)



    gene_num = data_filter.shape[1]

    return data, data1, data2, gene_num, len(results), label_dirc


def run(args):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.local_rank = torch.distributed.get_rank()
    # torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
    # torch.cuda.set_device(args.local_rank)
    global device
    device = torch.device("cuda")
    torch.autograd.set_detect_anomaly = True

    datasets = [
             'yan', 'Limb_Muscle', 'hrvatin', 'Tosches_turtle', 'Bach', 'deng','pollen', 'Diaphragm', 'Bladder', 'kolodziejczyk', 'Mammary_Gland', 'Adam', 'Klein', 'Quake_Smart-seq2_Trachea', 'muraro',
             'Romanov', 'Quake_Smart-seq2_Diaphragm', 'Quake_Smart-seq2_Heart', 'Quake_Smart-seq2_Limb_Muscle', 'Quake_Smart-seq2_Lung', 'Wang_Lung',
             'Quake_10x_Spleen', 'Quake_10x_Trachea',
             'Plasschaert', 'Young',
              'Chen'
                ]

    for data in datasets:
        print("Clustering in the dataset: ", data)
        args.dataname = data
        set_global_random_seed(args.seed)
        args.resPath, args.tensorboard = setup_path(args)

        # dataset loader
        torch_dataset, torch_dataset1, torch_dataset2, dim, num_classes, label_dirc = load_geometric_data(args.dataname)

        # train_loader = util_data.DataLoader(torch_dataset, batch_size=args.batch_size, shuffle=True,
        #                                     pin_memory=True)

        args.num_classes = num_classes
        out_dim = 128
        # model
        encode = encoder(args, dim, out_dim).to(device)

        encode_embedding = encode(torch_dataset.x.to(device), torch_dataset.edge_index.to(device))

        cluster_centers, clusterscores = get_confusion(encode_embedding, torch_dataset.y, num_classes, args.seed)

        print("cluster_centers.shape", cluster_centers.shape)

        model = Model(args, out_dim, cluster_centers=cluster_centers, alpha=1.0, encoder=encode).to(device)

        if torch.cuda.device_count() > 1:
            print("We have ", torch.cuda.device_count(), " GPUs!")

            model = torch.nn.parallel.DataParallel(model)

        print('Is model on gpu: ', next(model.parameters()).is_cuda)
        n_p = sum(x.nelement() for x in model.parameters())
        print(f"Number of parameter: {n_p/1e6:.2f}M")
        # optimizer
        optimizer = torch.optim.Adam([
            {'params': model.contrast_head.parameters(), 'lr': args.lr * 10},
            {'params': model.cluster_centers, 'lr': args.lr * 100, "weight_decay": 1e-3},
            {'params': encode.parameters(), 'lr': args.lr, "weight_decay": 1e-3}
        ], lr=args.lr)



        sample_num = len(torch_dataset.y)

        record_loss = {'cluster_loss': [], 'contrastive_loss': [], 'reg_loss': [], 'selfexpress_loss': []}
        record_cscore = {'ARI': [], 'NMI': [], 'AMI': []}
        ARI_flag = 0

        for it in range(200):
            model.train()
            encode.train()
            print("train step: ", it)
            x, edge, y = torch_dataset.x.to(device), torch_dataset.edge_index.to(device), torch_dataset.y.to(device)
            x1, edge1 = torch_dataset1.x.to(device), torch_dataset1.edge_index.to(device)
            x2, edge2 = torch_dataset2.x.to(device), torch_dataset2.edge_index.to(device)
            contrastive_encoder_emb = encode(x, edge)
            contrastive_encoder_emb1 = encode(x1, edge1)
            contrastive_encoder_emb2 = encode(x2, edge2)

            output_fin = model.get_cluster_prob(contrastive_encoder_emb, model.cluster_centers)
            labels_fin = output_fin.argmax(1)

            comb_central_emb_fin = torch.cat([contrastive_encoder_emb, model.cluster_centers], dim=0)
            feat_fin = F.normalize(model(comb_central_emb_fin), dim=1)
            feat1_fin, central_fin = feat_fin[: sample_num], feat_fin[sample_num:]

            feat1_fin1 = F.normalize(model(contrastive_encoder_emb1), dim=1)
            feat1_fin2 = F.normalize(model(contrastive_encoder_emb2), dim=1)

            losses = central_PairConLoss(temperature=args.temperature).cuda()(feat1_fin, labels_fin, central_fin)
            loss_pair = PairConLoss(temperature=args.temperature).cuda()(feat1_fin1, feat1_fin2, 0)
            loss = 0.5 * losses["loss"] + loss_pair["loss"]
            print("central loss: ", 0.5 * losses["loss"])
            print("pair loss: ",  loss_pair["loss"])

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()



            if (args.print_freq > 0) and ((it % args.print_freq == 0) or (it == args.max_iter)):
                model.eval()
                encode.eval()
                x, edge, y = torch_dataset.x.to(device), torch_dataset.edge_index.to(device), torch_dataset.y.to(device)
                contrastive_encoder_emb = encode(x, edge)

                embedding_list = contrastive_encoder_emb
                target_list = y.cpu().detach()
                print("The shape of embeddding in the evaluate step", embedding_list.shape, embedding_list.device)

                # get result at a Kmeans

                clustering_model = KMeans(n_clusters=num_classes, random_state=args.seed)
                clustering_model.fit(embedding_list.cpu().detach())
                embedding_kmeans_labels = torch.tensor(clustering_model.labels_).unsqueeze(1)

                embedding_confusion, embedding_confusion_centers = Confusion(args.num_classes), Confusion(
                    args.num_classes)
                # clustering accuracy
                embedding_confusion.add(embedding_kmeans_labels, target_list)
                embedding_confusion.optimal_assignment(args.num_classes)
                embedding_acc = embedding_confusion.acc()


                output = model.get_cluster_prob(embedding_list, model.cluster_centers)
                embedding_cluster_labels = output.argmax(1)

                embedding_confusion_centers.add(embedding_cluster_labels, target_list)
                embedding_confusion_centers.optimal_assignment(args.num_classes)
                embedding_centers_acc = embedding_confusion_centers.acc()

                kmeans_pre_labels, pre_labels, all_labels, embeddings, kmeans_score, model_score, acc, model_acc = embedding_kmeans_labels.cpu().detach().numpy(), embedding_cluster_labels.cpu().detach().numpy(), target_list.cpu().detach().numpy(), embedding_list.cpu().detach().numpy(), \
               embedding_confusion.clusterscores(), embedding_confusion_centers.clusterscores(), embedding_acc, embedding_centers_acc

                record_cscore["ARI"].append(model_score["ARI"])
                record_cscore["NMI"].append(model_score["NMI"])
                record_cscore["AMI"].append(model_score["AMI"])
                ARI = model_score["ARI"]
                if ARI > ARI_flag:
                    ARI_flag = ARI
                    np.save(args.resPath + 'best_central.npy',
                            model.cluster_centers.cpu().detach().numpy())
                    np.save(args.resPath + 'best_embedding.npy', embeddings)
                    np.save(args.resPath + 'Target_labels.npy', all_labels)
                    df = pd.DataFrame(
                        ["ARI:", kmeans_score["ARI"], " NMI:", kmeans_score["NMI"], " AMI:", kmeans_score["AMI"],
                         "ACC: ", acc])
                    df.to_csv(args.resPath + "_Representation_Clustering_bestscores.txt", sep=' ', index=False,
                              mode='a',
                              header=False)

                    df = pd.DataFrame(
                        ["ARI:", model_score["ARI"], " NMI:", model_score["NMI"], " AMI:", model_score["AMI"], "ACC: ",
                         model_acc])
                    df.to_csv(args.resPath + "_Representation_model_Clustering_bestscores.txt", sep=' ',
                              index=False, mode='a',
                              header=False)

                    f = create_img.create_img(embeddings, all_labels, label_dirc)
                    f.savefig(args.resPath + 'True_label_embedding_{}.jpg'.format(it))

                    f = create_img.create_img(embeddings, pre_labels, label_dirc)
                    f.savefig(args.resPath + 'pre_labels_embedding_{}.jpg'.format(it))


    return None


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0, 1, 2, 3],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=10, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')

    parser.add_argument('--which_contrastive', type=str, default='central_PairConLoss',
                        choices=["SingleCirculatePairConLoss", "PairConLoss", "DRC_contrastive"
                                 , "CirculatePairConLoss", "NT_Xent", "Circulate_NT_Xent", "Neighbor_PairConLoss",
                                 "central_PairConLoss", "HypersphereLoss"])

    # Dataset
    parser.add_argument('--datapath', type=str, default='../datasets/')
    parser.add_argument('--dataname', type=str, default='pollen', help="")
    parser.add_argument('--num_classes', type=int, default=2, help="")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--aug_prob', type=float, default=0.2, help="The prob of doing augmentation")
    parser.add_argument('--augmentation_type', type=str, default='noise', choices=["random", "top", "noise", "False"])
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')

    # Learning parameters
    # parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--encoder_type', type=str, default='Transformer', choices=["Transformer", "DNN"])
    parser.add_argument('--DNN_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--att_head', type=int, default=2)
    parser.add_argument('--v_dim', type=int, default=512)
    parser.add_argument('--k_dim', type=int, default=512)
    parser.add_argument('--fed_dim', type=int, default=1024)
    parser.add_argument('--att_layer', type=int, default=4)
    parser.add_argument('--reduce_dim', type=int, default=512)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')

    # contrastive learning
    parser.add_argument('--objective', type=str, default='central_contrastive', choices=["SCCL", "contrastive", "clustering",
                                                                                        "DRC_contrastive", "VAE_contrastive",
                                                                                        "ScName_contrastive", "central_contrastive",
                                                                                         "Hypersphere"])
    parser.add_argument('--change', type=str, default='GNN_central_label', help="decript the model in this training step")
    parser.add_argument('--batch_size', type=int, default=6000)
    parser.add_argument('--temperature', type=float, default=0.2, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=0.1, help="")

    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)

    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args


if __name__ == '__main__':
    import subprocess

    args = get_args(sys.argv[1:])

    if args.train_instance == "sagemaker":
        run(args)
        subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_resdir])
    else:
        run(args)






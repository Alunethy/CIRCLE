"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""
import torch.distributed as dist
from torch_kmeans import KMeans
# from sklearn.cluster import KMeans
import torch.nn.functional as F
import create_img
import numpy as np
from utils.logger import statistics_log
from utils.metric import Confusion
import pandas as pd
import torch
import torch.nn as nn
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss,InfoNCELoss
from learner.contrastive_utils import CirculatePairConLoss, SingleCirculatePairConLoss, SupConLoss, HypersphereLoss
from learner.contrastive_utils import NT_Xent, Circulate_NT_Xent, Neighbor_PairConLoss, central_PairConLoss
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import gc

class SCCLvTrainer(nn.Module):
    def __init__(self, model, trans, dim, optimizer, train_loader, args, label_dirc, which_contrastive, data_num, device):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.trans = trans
        self.optimizer = optimizer
        self.which_contrastive = which_contrastive
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        self.ARI_flag = 0.0
        self.project_ARI_flag = 0.0
        self.label_dirc = label_dirc
        self.dim = dim
        self.device = device

        self.cluster_loss = nn.KLDivLoss(size_average=False)

        if self.which_contrastive == 'PairConLoss':
            self.contrast_loss = PairConLoss(temperature=self.args.temperature).cuda()
        elif self.which_contrastive == 'infoNCE':
            self.contrast_loss=InfoNCELoss(temperature=self.args.temperature).cuda()
        elif self.which_contrastive == 'CirculatePairConLoss':
            self.contrast_loss = CirculatePairConLoss(temperature=self.args.temperature).cuda()
        elif self.which_contrastive == 'NT_Xent':
            self.contrast_loss = NT_Xent(data_num, self.args.temperature, 1).cuda()
        elif self.which_contrastive == 'Circulate_NT_Xent':
            self.contrast_loss = Circulate_NT_Xent(data_num, self.args.temperature, 1).cuda()
        elif self.which_contrastive == 'SingleCirculatePairConLoss':
            self.contrast_loss = SingleCirculatePairConLoss(temperature=self.args.temperature).cuda()
        elif self.which_contrastive == 'Neighbor_PairConLoss':
            self.contrast_loss = Neighbor_PairConLoss(temperature=self.args.temperature).cuda()
        elif self.which_contrastive == 'central_PairConLoss':
            self.contrast_loss = central_PairConLoss(temperature=self.args.temperature).cuda()
        elif self.which_contrastive == 'Sup_contrastive':
            self.contrast_loss = SupConLoss().cuda()
        elif self.which_contrastive == 'HypersphereLoss':
            self.contrast_loss = HypersphereLoss().cuda()
        else:
            print("there are some mistakes!")
        
        if self.args.objective=='ablation':
            print('ablation test!')
        if self.args.which_contrastive=='infoNCE':
            print('use infoNCE!')

        self.gstep = 0
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

    def train(self):
        print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()
        self.trans.train()
        record_loss = {'cluster_loss': [], 'contrastive_loss': [], 'reg_loss': [], 'selfexpress_loss': []}
        record_cscore = {'ARI': [], 'NMI': [], 'AMI': []}

        # init Kmeans
        #kmeans = KMeans(n_clusters=self.args.num_classes * 2, random_state=self.args.seed)

        for i in np.arange(self.args.max_iter+1):
            # for batch in self.train_loader:

            for t, batch in enumerate(self.train_loader):
                dataset, dataset2, dataset3 = batch['text'].to(self.device, non_blocking=True),\
                                              batch['augmentation_1'].to(self.device, non_blocking=True),\
                                                batch['augmentation_2'].to(self.device, non_blocking=True)
                # dataset = batch['text'].to(self.device)
                # get embedding

                # emb = self.trans.forward(dataset)
                # emb2 = self.trans.forward(dataset2)
                # emb3 = self.trans.forward(dataset3)

                # if t == 0:
                #     all_emb = emb.detach()
                #     all_emb2 = emb2.detach()
                #     all_emb3 = emb3.detach()
                # else:
                #     all_emb = torch.cat((all_emb, emb.detach()), dim=0)
                #     all_emb2 = torch.cat((all_emb2, emb2.detach()), dim=0)
                #     all_emb3 = torch.cat((all_emb3, emb3.detach()), dim=0)

                losses = self.train_step(dataset, dataset2, dataset3)
                record_loss["contrastive_loss"].append(losses['loss'].cpu().detach().numpy())
                record_loss["cluster_loss"].append(losses['cluster_loss'])

            if (self.args.print_freq > 0) and ((i % self.args.print_freq == 0) or (i == self.args.max_iter)):
                # statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                torch.distributed.barrier()

                kmeans_pre_labels, pre_labels, all_labels, embeddings, kmeans_score, model_score, acc, model_acc = self.evaluate_embedding(i, self.train_loader)

                record_cscore["ARI"].append(model_score["ARI"])
                record_cscore["NMI"].append(model_score["NMI"])
                record_cscore["AMI"].append(model_score["AMI"])
                ARI = model_score["ARI"]
                if ARI > self.ARI_flag:
                    self.ARI_flag = ARI
                    np.save(self.args.resPath + 'best_central.npy', self.model.module.cluster_centers.cpu().detach().numpy())
                    np.save(self.args.resPath + 'best_embedding.npy', embeddings)
                    np.save(self.args.resPath + 'Target_labels.npy', all_labels)
                    df = pd.DataFrame(["ARI:", kmeans_score["ARI"], " NMI:", kmeans_score["NMI"], " AMI:", kmeans_score["AMI"], "ACC: ", acc])
                    df.to_csv(self.args.resPath + "_Representation_Clustering_bestscores.txt", sep=' ', index=False,
                              header=False)

                    df = pd.DataFrame(
                        ["ARI:", model_score["ARI"], " NMI:", model_score["NMI"], " AMI:", model_score["AMI"], "ACC: ", model_acc])
                    df.to_csv(self.args.resPath + "_Representation_model_Clustering_bestscores.txt", sep=' ', index=False,
                              header=False)
                    
                    # disable mid-process tsne to save time
                    # f = create_img.create_img(embeddings, all_labels, self.label_dirc)
                    # f.savefig(self.args.resPath + 'True_label_embedding_{}.jpg'.format(i))

                    # f = create_img.create_img(embeddings, pre_labels, self.label_dirc)
                    # f.savefig(self.args.resPath + 'pre_labels_embedding_{}.jpg'.format(i))

                self.model.train()

        record_x1 = [i for i in range(len(record_loss['cluster_loss']))]
        record_x2 = [i for i in range(len(record_cscore["ARI"]))]
        g = create_img.line_maker(record_x1, record_x2, record_loss, record_cscore, self.args)
        g.savefig(self.args.resPath + 'summary.jpg')
        return None

    def reparameterize(self, x):
        c_dim = x.shape[1]
        z_dim = c_dim // 2
        c_mu = x[:, :z_dim]
        c_log_var = x[:, z_dim: z_dim * 2]
        z_signal = torch.randn(c_mu.size()).to(self.device, non_blocking=True)
        z_c = c_mu + torch.exp(c_log_var / 2) * z_signal
        return z_c, z_signal

    def select_contrastive(self, temp, threshold):
        temp = torch.nn.Softmax(dim=1)(temp)
        max_value, max_index = torch.max(temp, dim=1)
        select = []
        for i in range(len(max_value)):
            if max_value[i] > threshold:
                select.append(i)

        return select

    def train_step(self, dataset, dataset2, dataset3):
        # Clustering loss and Instance-CL loss

        # have used the cluster_centers
        if self.args.objective == "SCCL":
            embd = self.trans.forward(dataset)
            embd2 = self.trans.forward(dataset2)

            output = self.model.module.get_cluster_prob(embd)  # get student`s t-distribution
            print("output.shape", output.shape)
            target = target_distribution(output).detach()  # get target distribution
            print("target.shape", target.shape)

            output2 = self.model.module.get_cluster_prob(embd2)  # get student`s t-distribution
            target2 = target_distribution(output2).detach()  # get target distribution

            cluster_loss = (self.cluster_loss((output + 1e-08).log(), target2) / output.shape[0]) \
                           + (self.cluster_loss((output + 1e-08).log(), target) / output.shape[0])
                           # + (self.cluster_loss((output + 1e-08).log(), target3) / output.shape[0]) \

            feat1 = F.normalize(self.model(embd), dim=1)
            feat2 = F.normalize(self.model(embd2), dim=1)
            # losses = self.contrast_loss(feat1, feat2, output.max(1)[1])
            losses = self.contrast_loss(feat1, feat2, output.argmax(1))
            losses_class = self.contrast_loss(output.T, output2.T, output.argmax(1))

            DRC_cluster_loss = self.DRC_ClusterLoss(output, output2)

            loss = cluster_loss.item() + losses["loss"] + DRC_cluster_loss.item() + losses_class["loss"]
            losses["cluster_loss"] = cluster_loss.item() + DRC_cluster_loss.item()
            losses["loss"] = losses["loss"] + losses_class["loss"]
        elif self.args.objective=='ablation':
            num=dataset.shape[0]
            z2=self.trans.forward(dataset2)
            z3=self.trans.forward(dataset3)
            zz = torch.cat([z2, z3], dim=0)
            ff = F.normalize(self.model.forward(zz), dim=1)
            f2, f3 = ff[: num], ff[num:]
            losses=self.contrast_loss(f2, f3)
            loss = losses["loss"]
            losses["cluster_loss"] = 0

        elif self.args.objective == "Hypersphere":
            embd = self.trans.forward(dataset)
            embd_BN = embd.shape[0]
            embd2 = self.trans.forward(dataset2)

            comb_central_emb = torch.cat([embd, embd2, self.model.module.cluster_centers], dim=0)
            feat = F.normalize(self.model(comb_central_emb), dim=1)
            # embd2 = self.trans.forward(dataset2)
            feat1, feat2, central = feat[: embd_BN], feat[embd_BN: 2 * embd_BN], feat[2 * embd_BN:]

            # losses = self.contrast_loss(feat1, feat2, output.max(1)[1])
            losses = self.contrast_loss(feat1, feat2)

            print("Hypersphere[loss]", losses["loss"])

            loss = losses["loss"]
            losses["cluster_loss"] = 0

        elif self.args.objective == "VAE_contrastive":
            embd = self.trans.forward(dataset)
            # embd2 = self.trans.forward(dataset2)
            # get vae
            latents1, z_normal1 = self.reparameterize(embd)
            latents2, z_normal2 = self.reparameterize(embd)

            # feat1, feat2 = self.model.contrast_logits(embd2, embd3)
            feat1 = F.normalize(self.model(latents1), dim=1)
            feat2 = F.normalize(self.model(latents2), dim=1)
            # losses = self.contrast_loss(feat1, feat2, output.max(1)[1])
            losses = self.contrast_loss(feat1, feat2, 0)
            kl_div_loss1 = F.kl_div(latents1.softmax(dim=-1).log(), z_normal1.softmax(dim=-1), reduction='mean')
            kl_div_loss2 = F.kl_div(latents2.softmax(dim=-1).log(), z_normal2.softmax(dim=-1), reduction='mean')

            loss = losses["loss"] + kl_div_loss1 + kl_div_loss2
            losses["cluster_loss"] = kl_div_loss1 + kl_div_loss2


        # Instance-CL loss only
        elif self.args.objective == "contrastive":
            # embeddings = embd.cpu().numpy()
            # pred_labels = torch.tensor(kmeans.labels_.astype(np.int)).squeeze(0)
            embd_BN = dataset.shape[0]

            # comb_encoder_emb_fin = torch.cat([dataset, dataset2], dim=0)
            # comb_encoder_emb_fin = self.trans.forward(comb_encoder_emb_fin)
            # embd_fin, embd_2 = comb_encoder_emb_fin[: embd_BN], comb_encoder_emb_fin[embd_BN:]
            embd_fin = self.trans.forward(dataset)
            embd_2 = self.trans.forward(dataset2)
            output_fin = self.model.module.get_cluster_prob(embd_fin, self.model.module.cluster_centers)

            # target = target_distribution(output_fin).detach()  # get target distribution
            # cluster_loss = self.cluster_loss((output_fin + 1e-08).log(), target)

            labels_fin = output_fin.argmax(1)

            comb_central_emb_fin = torch.cat([embd_fin, embd_2, self.model.module.cluster_centers], dim=0)
            feat_fin = F.normalize(self.model.forward(comb_central_emb_fin), dim=1)
            feat1_fin, feat2_fin, central_fin = feat_fin[: embd_BN], feat_fin[embd_BN: 2 * embd_BN], feat_fin[
                                                                                                     2 * embd_BN:]

            # feat1_fin = self.model.module.forward(embd_fin)
            # feat2_fin = self.model.forward(embd_2)
            # central_fin = self.model.module.forward(self.model.module.cluster_centers)

            losses = self.contrast_loss(feat1_fin, feat2_fin, labels_fin)
            # loss_pair = PairConLoss(temperature=self.args.temperature).cuda()(feat1_fin, feat2_fin, 0)
            # losses_tem = PairConLoss(temperature=self.args.temperature).cuda()(central_fin, central_fin, 0)

            # use the losses and loss_pair is not good
            # only use losses that could get a great result

            loss = losses["loss"]
            print("losses[loss]:", losses["loss"])
            losses["cluster_loss"] = 0

        elif self.args.objective == "Top_contrastive":
            embd_BN = dataset.shape[0]

            embd_fin = self.trans.forward(dataset)
            # embd_2 = self.trans.forward(dataset2)
            output_fin = self.model.module.get_cluster_prob(embd_fin, self.model.module.cluster_centers)
            labels_fin = output_fin.argmax(1)
            
            print("embd_fin", embd_fin.shape)
            print("self.model.module.cluster_centers", self.model.module.cluster_centers.shape)

            # index = self.select_contrastive(output_fin, 0.4)
            max_value, max_index = torch.max(output_fin, dim=1)
            values, indices = max_value.topk(int(embd_BN / self.args.num_classes), dim=0, largest=True, sorted=False)
            # index = max_index[indices]

            if len(indices) > 0:
                print("index", indices)
                feat = self.model.forward(embd_fin[indices])
                centers = self.model.forward(self.model.module.cluster_centers)
                print("centers", centers.shape)
                print("feat", feat.shape)

                feat1_fin = F.normalize(feat, dim=1)
                central_fin = F.normalize(centers, dim=1)
                print("feat1_fin", feat1_fin)
                print("central_fin", central_fin)
                # comb_central_emb_fin = torch.cat([embd_fin, self.model.module.cluster_centers], dim=0)
                # feat_fin = F.normalize(self.model.forward(comb_central_emb_fin), dim=1)
                # feat1_fin, central_fin = feat_fin[: embd_BN][index], feat_fin[embd_BN: ]


                losses = central_PairConLoss().cuda()(feat, labels_fin[indices], centers)
                # feature = torch.cat([feat1_fin.view(len(index), 1, -1), feat2_fin.view(len(index), 1, -1)], dim=1)
                # losses = SupConLoss().cuda()(feature)
                # losses_top = SupConLoss().cuda()(feature, labels_fin[index])

                loss = losses["loss"]
                print("[loss]:", loss)
                losses["cluster_loss"] = 0

            else:
                # comb_central_emb_fin = torch.cat([embd_fin, self.model.module.cluster_centers], dim=0)
                # feat_fin = F.normalize(self.model.forward(comb_central_emb_fin), dim=1)
                # feat1_fin, central_fin = feat_fin[: embd_BN], feat_fin[embd_BN: ]
                feat = self.model.forward(embd_fin)
                centers = self.model.forward(self.model.module.cluster_centers)
                feat1_fin = F.normalize(feat, dim=1)
                central_fin = F.normalize(centers, dim=1)


                losses = central_PairConLoss().cuda()(feat1_fin, labels_fin, central_fin)

                # feature = torch.cat([feat1_fin.view(embd_BN, 1, -1), feat2_fin.view(embd_BN, 1, -1)], dim=1)
                # losses = PairConLoss().cuda()(feat1_fin, feat2_fin)
                # losses_top = SupConLoss().cuda()(feature, labels_fin)
                # losses = SupConLoss().cuda()(feature)

                loss = losses["loss"]
                print("[loss]:", loss)
                losses["cluster_loss"] = 0


            # if len(select) > 0:
            #     embd_BN = len(select)
            #     # comb_central_emb_fin = torch.cat([embd_fin[select], embd_2[select]], dim=0)
            #     # feat_fin = F.normalize(self.model.forward(comb_central_emb_fin), dim=1)
            #     feat1_fin, feat2_fin = feat_fin[: embd_BN][select], feat_fin[embd_BN:][select]
            #
            #     losses_top = central_PairConLoss().cuda()((feat1_fin + feat2_fin)/2, labels_fin[select], central_fin)
            #
            #     loss = losses["loss"] + losses_top["loss"]
            #     print("[loss]:", loss)
            #     losses["cluster_loss"] = 0
            # else:
            #     loss = losses["loss"]
            #     print("[loss]:", loss)
            #     losses["cluster_loss"] = 0

        elif self.args.objective == "Sup_contrastive":
            embd_BN = dataset.shape[0]

            embd_fin = self.trans.forward(dataset)
            embd_2 = self.trans.forward(dataset2)
            output_fin = self.model.module.get_cluster_prob(embd_fin, self.model.module.cluster_centers)

            labels_fin = output_fin.argmax(1)

            comb_central_emb_fin = torch.cat([embd_fin, embd_2], dim=0)
            feat_fin = F.normalize(self.model.forward(comb_central_emb_fin), dim=1)
            feat1_fin, feat2_fin = feat_fin[: embd_BN].view(embd_BN, 1, -1), feat_fin[embd_BN:].view(embd_BN, 1, -1)

            feature = torch.cat([feat1_fin, feat2_fin], dim=1)
            losses = SupConLoss().cuda()(feature, labels_fin)

            loss = losses["loss"]
            print("losses[loss]:", losses["loss"])
            losses["cluster_loss"] = 0

        elif self.args.objective == "central_contrastive":
            # point and central
            embd_BN = dataset.shape[0]

            embd_fin = self.trans.forward(dataset)
            # embd_2 = self.trans.forward(dataset2)
            output_fin = self.model.module.get_cluster_prob(embd_fin, self.model.module.cluster_centers)

            labels_fin = output_fin.argmax(1)
            select = self.select_contrastive(output_fin, 0.5)

            comb_central_emb_fin = torch.cat([embd_fin, self.model.module.cluster_centers], dim=0)
            feat_fin = F.normalize(self.model.forward(comb_central_emb_fin), dim=1)
            feat1_fin, central_fin = feat_fin[: embd_BN], feat_fin[embd_BN:]

            losses = self.contrast_loss(feat1_fin[select], labels_fin[select], central_fin)

            loss = losses["loss"]
            print("losses[loss]:", losses["loss"])
            losses["cluster_loss"] = 0

        # Clustering loss only

        elif self.args.objective == "clustering":
            embd_BN = dataset.shape[0]

            comb_encoder_emb_fin = torch.cat([dataset, dataset2], dim=0)
            comb_encoder_emb_fin = self.trans.forward(comb_encoder_emb_fin)
            embd_fin, embd_2 = comb_encoder_emb_fin[: embd_BN], comb_encoder_emb_fin[embd_BN:]
            output_fin = self.model.module.get_cluster_prob(embd_fin, self.model.module.cluster_centers) # get student`s t-distribution

            target = target_distribution(output_fin).detach()  # get target distribution

            labels_fin = output_fin.argmax(1)

            comb_central_emb_fin = torch.cat([embd_fin, embd_2, self.model.module.cluster_centers], dim=0)
            feat_fin = F.normalize(self.model(comb_central_emb_fin), dim=1)
            feat1_fin, feat2_fin, central_fin = feat_fin[: embd_BN], feat_fin[embd_BN: 2 * embd_BN], feat_fin[
                                                                                                     2 * embd_BN:]

            output2 = self.model.module.get_cluster_prob(embd_2, self.model.module.cluster_centers)  # get student`s t-distribution
            target2 = target_distribution(output2).detach()  # get target distribution

            cluster_loss = (self.cluster_loss((output_fin + 1e-08).log(), target) / output_fin.shape[0]) \
                           + (self.cluster_loss((output2 + 1e-08).log(), target2) / output_fin.shape[0])


            losses = self.contrast_loss(feat1_fin, feat2_fin, labels_fin)

            loss = cluster_loss.item() + losses["loss"]
            losses["cluster_loss"] = cluster_loss.item()

        else:
            print("maybe mistake!")
            loss = None
            losses = None

        self.optimizer.zero_grad()
        loss.sum().backward()
        self.optimizer.step()

        return losses


    def evaluate_embedding(self, step, dataloader):
        print('---- {} evaluation batches ----'.format(len(dataloader)))
        embedding_list = []
        target_list = []

        self.model.eval()
        self.trans.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch["text"].to(self.device, non_blocking=True), \
                              batch["label"].to(self.device, non_blocking=True)

                embeddings = self.trans.forward(text)

                batch_pred = [torch.zeros_like(embeddings) for _ in range(dist.get_world_size())]
                dist.all_gather(batch_pred, embeddings)
                embedding_list.extend(batch_pred)

                batch_pred = [torch.zeros_like(label) for _ in range(dist.get_world_size())]
                dist.all_gather(batch_pred, label)
                target_list.extend(batch_pred)


        # Initialize confusion matrices

        embedding_list = torch.cat(embedding_list, 0)
        target_list = torch.cat(target_list, 0)
        print("The shape of embeddding in the evaluate step", embedding_list.shape, embedding_list.device)

        # get result at a Kmeans
        gc.collect()
        kmeans_evaluate = KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
        embedding_kmeans_labels = torch.tensor(kmeans_evaluate.fit_predict(embedding_list.unsqueeze(0).cpu())).squeeze(0)
        # clustering_model = KMeans(n_clusters=self.args.num_classes, random_state=self.args.seed)
        # clustering_model.fit(embedding_list.cpu())
        # embedding_kmeans_labels = clustering_model.labels_

        embedding_confusion, embedding_confusion_centers = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        # clustering accuracy
        embedding_confusion.add(embedding_kmeans_labels, target_list)
        embedding_confusion.optimal_assignment(self.args.num_classes)
        embedding_acc = embedding_confusion.acc()

        ressave = {"acc": embedding_acc}
        ressave.update(embedding_confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        output = self.model.module.get_cluster_prob(embedding_list, self.model.module.cluster_centers)
        embedding_cluster_labels = output.argmax(1)

        embedding_confusion_centers.add(embedding_cluster_labels, target_list)
        embedding_confusion_centers.optimal_assignment(self.args.num_classes)
        embedding_centers_acc = embedding_confusion_centers.acc()

        ##########################################################

        gc.collect()
        return embedding_kmeans_labels.cpu().numpy(), embedding_cluster_labels.cpu().numpy(), target_list.cpu().numpy(), embedding_list.cpu().numpy(), \
               embedding_confusion.clusterscores(), embedding_confusion_centers.clusterscores(), embedding_acc, embedding_centers_acc


if __name__ == '__main__':
    embedding = nn.Embedding(10, 5)  # 10个词，每个词用2维词向量表示
    input = torch.arange(0, 6).view(3, 2).long()  # 3个句子，每句子有2个词
    input = torch.repeat_interleave(input, 3, dim=0)
    print(input)
    tex = torch.randn(9, 2).unsqueeze(1).transpose(1, 2)
    print(tex.shape)
    output = embedding(input)
    print(output.size())

    rese = output * tex
    print(rese)
    print(rese.shape)

    fin = rese.sum(1)
    print(fin)
    print(fin.shape)

             
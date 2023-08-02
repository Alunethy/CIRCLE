"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""
import torch.distributed as dist
import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
import pandas as pd
import gc
import create_img

def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    return token_feat

def get_confusion(embedding, labels, num_classes, seed):
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes, random_state=seed)
    clustering_model.fit(embedding.cpu().detach())
    cluster_assignment = clustering_model.labels_

    true_labels = labels
    pred_labels = torch.tensor(cluster_assignment)
    # print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(embedding.shape, len(true_labels),
    #                                                                  len(pred_labels)))

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(),
                                                                    clustering_model.cluster_centers_.shape))
    print('Clustering scores:', confusion.clusterscores())
    gc.collect()
    return clustering_model.cluster_centers_, confusion.clusterscores()


def get_kmeans_centers(trans, train_loader, num_classes, args, device, resPath, label_dirc):
    pred_list = []
    target_list = []
    ori_list = []
    for i, batch in enumerate(train_loader):
        dataset, label = batch["text"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)

        embeddings = trans.forward(dataset)
        batch_pred = [torch.zeros_like(embeddings) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_pred, embeddings)
        pred_list.extend(batch_pred)

        batch_pred = [torch.zeros_like(label) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_pred, label)
        target_list.extend(batch_pred)

        batch_pred = [torch.zeros_like(dataset.squeeze(1)) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_pred, dataset.squeeze(1))
        ori_list.extend(batch_pred)

    pred_list = torch.cat(pred_list, 0).cpu()
    target_list = torch.cat(target_list, 0).cpu()
    ori_list = torch.cat(ori_list, 0).cpu()

    torch.cuda.empty_cache()
    cluster_centers, clusterscores = get_confusion(pred_list, target_list, num_classes, args.seed)
    print("Ori_Clustering scores as follow")
    ori_cluster_centers, ori_clusterscores = get_confusion(ori_list, target_list, num_classes, args.seed)
    df = pd.DataFrame(["ARI:", ori_clusterscores["ARI"], " NMI:", ori_clusterscores["NMI"], " AMI:", ori_clusterscores["AMI"]])
    df.to_csv(args.resPath + "_Ori_scores.txt", sep=' ', index=False, header=False)
    f = create_img.create_img(ori_list, target_list, label_dirc)
    f.savefig(resPath + 'Ori_kmeans.jpg')

    # confusion = Confusion(num_classes)
    # clustering_model = KMeans(n_clusters=num_classes)
    # clustering_model.fit(all_embeddings)
    # cluster_assignment = clustering_model.labels_
    #
    # true_labels = all_labels
    # pred_labels = torch.tensor(cluster_assignment)
    # print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels),
    #                                                                  len(pred_labels)))
    #
    # confusion.add(pred_labels, true_labels)
    # confusion.optimal_assignment(num_classes)
    # print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(),
    #                                                                 clustering_model.cluster_centers_.shape))

    return cluster_centers




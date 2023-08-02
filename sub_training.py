"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class SCCLBert(nn.Module):
    def __init__(self, args, dim, DNN_dim, cluster_centers=None, alpha=1.0, data_num=0, device="cuda"):
        super(SCCLBert, self).__init__()
        self.data_num = data_num
        self.device = device
        self.args = args

        self.alpha = alpha
        self.dim = dim
        if args.encoder_type == "DNN":
            input_dim = DNN_dim
        else:
            input_dim = self.dim

        if args.objective == "VAE_contrastive":
            input_dim = input_dim // 2

        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512)
        )

        # Clustering head

        self.emb_size = cluster_centers.shape[-1]
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers, requires_grad=True)

    def forward(self, data_batch):
        return self.contrast_head(data_batch)


    # def get_mean_embeddings(self, input_ids, attention_mask):
    #     bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
    #     attention_mask = attention_mask.unsqueeze(-1)
    #     mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
    #     return mean_output
    
    # the equation 3 of paper SCCL
    def get_cluster_prob(self, embeddings, cluster_centers):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - cluster_centers.to(self.device, non_blocking=True)) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2
    
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2!=None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

    def get_cluster_central_contras(self):
        feat1 = F.normalize(self.contrast_head(self.cluster_centers), dim=1)
        return feat1


if __name__ == '__main__':
    input = torch.rand(128, 1, 4096).cuda()
    d_model = input.shape[-1]
    # model = SCCLBert(args, dim, DNN_dim, cluster_centers=None, alpha=1.0, data_num=0, device="cuda").cuda()
    # n_p = sum(x.numel() for x in model.parameters())
    # print("model net par", n_p)
    # print("model net:", model)
    # trans_emb = model(input)
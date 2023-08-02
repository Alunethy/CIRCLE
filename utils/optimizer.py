"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch
# from transformers import AutoModel, AutoTokenizer, AutoConfig
# from sentence_transformers import SentenceTransformer

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}


def get_optimizer(model, trans, args):
    if args.objective == "SCCL":
        optimizer = torch.optim.Adam([
            {'params': model.contrast_head, 'lr': args.lr},
            {'params': model.cluster_centers, 'lr': args.lr * args.lr_scale, "weight_decay": 1e-3},
            {'params': trans.parameters(), 'lr': args.lr * args.lr_scale, "weight_decay": 1e-3}
        ], lr=args.lr)
    elif args.objective == "contrastive":
        optimizer = torch.optim.Adam([
            {'params': model.module.contrast_head.parameters(), 'lr': args.lr},
            {'params': model.module.cluster_centers, 'lr': args.lr * 10, "weight_decay": 1e-3},
            {'params': trans.parameters(), 'lr': args.lr * 10, "weight_decay": 1e-3}
        ], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.module.contrast_head.parameters(), 'lr': args.lr},
            {'params': model.module.cluster_centers, 'lr': args.lr * 10,  "weight_decay": 1e-3},
            {'params': trans.parameters(), 'lr': args.lr * 10, "weight_decay": 1e-3}
        ], lr=args.lr)

    print(optimizer)
    return optimizer 
    









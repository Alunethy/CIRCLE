"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import sys
sys.path.append('./')
import trans_model, load_data
import torch
import argparse
from training import SCCLvTrainer
from utils.Kmeans import get_kmeans_centers
from utils.logger import setup_path, set_global_random_seed
from utils.optimizer import get_optimizer
from sub_training import SCCLBert
import torch.utils.data as util_data
import torch.distributed as dist

def run(args):
    torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
    torch.cuda.set_device(args.local_rank)
    global device
    device = torch.device("cuda", args.local_rank)
    torch.autograd.set_detect_anomaly = True

    
    datasets=['Adam', 'Bach', 'Bladder', 'deng', 'Diaphragm', 'hrvatin', 'Klein', 'kolodziejczyk', 'Limb_Muscle', 
              'Mammary_Gland', 'muraro', 'Plasschaert', 'pollen', 'Quake_10x_Spleen', 'Quake_10x_Trachea', 'Quake_Smart-seq2_Diaphragm', 
              'Quake_Smart-seq2_Heart', 'Quake_Smart-seq2_Limb_Muscle', 'Quake_Smart-seq2_Lung', 'Quake_Smart-seq2_Trachea', 'Romanov', 
              'Tosches_turtle', 'Wang_Lung', 'yan', 'Young']

    for data in datasets:
        print("Clustering in the dataset: ", data)
        args.dataname = data
        set_global_random_seed(args.seed)
        args.resPath, args.tensorboard = setup_path(args)

        # dataset loader
        torch_dataset, dim, num_classes, label_dirc = load_data.load_data(args.dataname)
        data_num = torch_dataset.__len__()

        # train_loader = util_data.DataLoader(torch_dataset, batch_size=args.batch_size, shuffle=True,
        #                                     pin_memory=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(torch_dataset)
        train_dataloaders = util_data.DataLoader(torch_dataset, batch_size=args.batch_size, num_workers=4, sampler=train_sampler, shuffle=False,
                                       pin_memory=True)

        args.num_classes = num_classes

        # model
        trans = trans_model.Trans(d_model=dim, DNN_dim=args.DNN_dim, head=args.att_head, layer=args.att_layer, k_dim=args.k_dim,
                                  v_dim=args.v_dim, fed_dim=args.fed_dim, encoder_type=args.encoder_type)
        # trans = trans_model.Ori_Trans(d_model=dim, head=args.att_head, layer=args.att_layer, fed_dim=args.fed_dim)

        # trans = torch.nn.DataParallel(trans).cuda(0)
        #
        # # initialize cluster centers in the sub_training
        # print("compute cluster_centers")


        trans = trans.to(device)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
        trans = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trans)  # 设置多个gpu的BN同步
        trans = torch.nn.parallel.DistributedDataParallel(trans,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True,
                                                          broadcast_buffers=False)

        cluster_centers = get_kmeans_centers(trans, train_dataloaders, args.num_classes, args, device, args.resPath, label_dirc)
        print("cluster_centers.shape", cluster_centers.shape)

        model = SCCLBert(args, dim, DNN_dim=args.DNN_dim, cluster_centers=cluster_centers, alpha=args.alpha, data_num=data_num, device=device)

        if torch.cuda.device_count() > 1:
            print("We have ", torch.cuda.device_count(), " GPUs!")
            My_model = model.to(device)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
            My_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(My_model)  # 设置多个gpu的BN同步
            model = torch.nn.parallel.DistributedDataParallel(My_model,
                                                                 device_ids=[args.local_rank],
                                                                 output_device=args.local_rank,
                                                                 find_unused_parameters=True,
                                                                 broadcast_buffers=False)

        print('Is model on gpu: ', next(model.parameters()).is_cuda)
        n_p = sum(x.nelement() for x in model.parameters())
        print(f"Number of parameter: {n_p/1e6:.2f}M")
        # optimizer
        optimizer = get_optimizer(model, trans, args)
        # optimizer = torch.nn.DataParallel(optimizer).module

        trainer = SCCLvTrainer(model, trans, dim, optimizer, train_dataloaders, args, label_dirc, args.which_contrastive, data_num, device)
        print("Begin to train")
        trainer.train()
    return None


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0, 1, 2, 3],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=5, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')

    parser.add_argument('--which_contrastive', type=str, default='SupConLoss',
                        choices=["SingleCirculatePairConLoss", "PairConLoss", "DRC_contrastive"
                                 , "CirculatePairConLoss", "NT_Xent", "Circulate_NT_Xent", "Neighbor_PairConLoss",
                                 "central_PairConLoss", "HypersphereLoss", "SupConLoss",'infoNCE'])

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
    parser.add_argument('--max_iter', type=int, default=50)
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
    parser.add_argument('--objective', type=str, default='Top_contrastive', choices=["SCCL",'para','ablation',"contrastive", "clustering",
                                                                                        "DRC_contrastive", "VAE_contrastive",
                                                                                        "ScName_contrastive", "central_contrastive",
                                                                                         "Hypersphere", "Sup_contrastive", "Top_contrastive"])
    parser.add_argument('--change', type=str, default='threshold_t_label', help="decript the model in this training step")
    parser.add_argument('--batch_size', type=int, default=6000)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=0.1, help="")

    parser.add_argument('--lam', type=float, default=0.1, help="")

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






import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
# Parameters and DataLoaders
input_dim = 5000
output_size = 2

batch_size = 2000
data_size = 10000


torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
# 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)  # 设置gpu编号为local_rank;此句也可能看出local_rank的值是什么


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.label = torch.randint(0, 2, [self.len])

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

train_dataset = RandomDataset(input_dim, data_size)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler, pin_memory=True)

test_dataset = RandomDataset(input_dim, data_size)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
test_dataloaders = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=test_sampler, pin_memory=True)

# rand_loader = DataLoader(dataset=RandomDataset(input_dim, data_size),
#                          batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size, self.input_size))

    def forward(self, input):
        output = self.fc(input)

        return output

model = Model(input_dim, output_size)


# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
My_model = model.cuda(args.local_rank)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
My_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(My_model)  # 设置多个gpu的BN同步
My_model = torch.nn.parallel.DistributedDataParallel(My_model,
                                                       device_ids=[args.local_rank],
                                                       output_device=args.local_rank,
                                                       find_unused_parameters=False,
                                                       broadcast_buffers=False)


class Model2(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.output_size))

    def forward(self, input):
        output = self.fc(input)
        return output

model2 = Model2(input_dim, output_size)
model2 = model2.cuda(args.local_rank)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2)  # 设置多个gpu的BN同步
model2 = torch.nn.parallel.DistributedDataParallel(model2,
                                                       device_ids=[args.local_rank],
                                                       output_device=args.local_rank,
                                                       find_unused_parameters=False,
                                                       broadcast_buffers=False)

opt = torch.optim.Adam([
            {'params': My_model.parameters(), 'lr': 0.0001},
            {'params': model2.parameters(), 'lr': 0.005}
        ], lr=0.01)

# opt = torch.optim.Adam(My_model.parameters())
for epoch in range(20):
    train_sampler.set_epoch(epoch)  # 这句莫忘，否则相当于没有shuffle数据
    My_model.train()
    for idx, sample in enumerate(train_dataloaders):
        inputs, targets = sample[0].cuda(args.local_rank, non_blocking=True), sample[1].cuda(args.local_rank, non_blocking=True)
        opt.zero_grad()
        output = My_model.forward(inputs)
        output = model2.forward(output)
        loss = F.cross_entropy(output, targets)  #
        loss.backward()
        opt.step()
        loss = reduce_mean(loss, dist.get_world_size())
        print("loss:", loss)

torch.distributed.barrier()

# if args.local_rank == 0:
pred_list = []
target_list = []
My_model.eval()

with torch.no_grad():
    for idx, sample in enumerate(test_dataloaders):
        inputs, targets = sample[0].cuda(args.local_rank, non_blocking=True), sample[1].cuda(args.local_rank,
                                                                                             non_blocking=True)
        opt.zero_grad()
        output = My_model(inputs)
        batch_pred = [torch.zeros_like(output) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_pred, output)
        pred_list.extend(batch_pred)

        batch_pred = [torch.zeros_like(targets) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_pred, targets)
        target_list.extend(batch_pred)

pred_list = torch.cat(pred_list, 0)
print("pred_list", pred_list.shape)
target_list = torch.cat(target_list, 0)
print("target_list", target_list.shape)

pre = torch.max(pred_list, 1)[1]
acc = accuracy_score(y_true=target_list.cpu(), y_pred=pre.cpu())
print("acc", acc)
print("device", pre.device)



import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse

class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=1., device="cuda"):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.device = device
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape).to(self.device, non_blocking=True)
        self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0,1).sample(self.b_mu.shape).to(self.device, non_blocking=True)
        self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)


class MLP_BBB(nn.Module):
    def __init__(self, hidden_units, noise_tol=.1,  prior_var=1., device="cuda", class_num=0):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()
        self.class_num = class_num
        self.device = device
        self.hidden = Linear_BBB(hidden_units, 64, prior_var=prior_var, device=self.device)
        self.out = Linear_BBB(64, self.class_num, prior_var=prior_var, device=self.device)
        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):
        # again, this is equivalent to a standard multilayer perceptron
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

    def log_prior(self):
        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss




# toy dataset we can start with
input_dim = 5000
output_size = 2

batch_size = 500
data_size = 1000

class_num = 5


torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
# 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)  # 设置gpu编号为local_rank;此句也可能看出local_rank的值是什么
global device
if torch.cuda.device_count() > 1:
    device = torch.device("cuda", args.local_rank)
else:
    device = torch.device("cuda", 0)

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class RandomDataset(Dataset):

    def __init__(self, size, length, class_num):
        self.len = length
        self.data = torch.randn(length, size)
        self.label = torch.randint(0, class_num, [self.len])

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len



train_dataset = RandomDataset(input_dim, data_size, class_num)
test_dataset = RandomDataset(input_dim, data_size, class_num)
net = MLP_BBB(input_dim, prior_var=10, device=device, class_num=class_num)
if torch.cuda.device_count() > 1:
    print("We have ", torch.cuda.device_count(), "GPUs!")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler, pin_memory=True)


    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloaders = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, sampler=test_sampler, pin_memory=True)



    net = net.cuda(args.local_rank)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # 设置多个gpu的BN同步
    net = torch.nn.parallel.DistributedDataParallel(net,
                                                           device_ids=[args.local_rank],
                                                           output_device=args.local_rank,
                                                           find_unused_parameters=False,
                                                           broadcast_buffers=False)
else:
    train_dataloaders = DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                   pin_memory=True)
    test_dataloaders = DataLoader(test_dataset, batch_size=batch_size, num_workers=4,
                                  pin_memory=True)
    net.cuda()

opt = optim.Adam(net.parameters(), lr=.1)
epochs = 1000
for epoch in range(epochs):  # loop over the dataset multiple times
    for idx, sample in enumerate(train_dataloaders):
        inputs, targets = sample[0].to(device, non_blocking=True), sample[1].to(device, non_blocking=True)
        opt.zero_grad()
        # forward + backward + optimize
        # loss = net.sample_elbo(inputs, targets, 1)
        input, target, samples = inputs, targets, class_num

        outputs = torch.zeros(samples, target.shape[0]).to(device, non_blocking=True)
        log_priors = torch.zeros(samples).to(device, non_blocking=True)
        log_posts = torch.zeros(samples).to(device, non_blocking=True)
        log_likes = torch.zeros(samples).to(device, non_blocking=True)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            out = net(input).reshape(-1, class_num)  # make predictions
            print("outputs[i].shape", outputs[i].shape)
            print("out.shape", out.shape)
            outputs[i] = out
            log_priors[i] = net.module.log_prior()  # get log prior
            log_posts[i] = net.module.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], net.module.noise_tol).log_prob(
                target.reshape(-1)).sum()  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like

        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            loss = reduce_mean(loss, dist.get_world_size())
            print('epoch: {}/{}'.format(epoch+1, epochs))
            print('Loss:', loss.item())
print('Finished Training')


# samples is the number of "predictions" we make for 1 x-value.
# samples = 100
# x_tmp = torch.linspace(-5,5,100).reshape(-1,1)
# y_samp = np.zeros((samples,100))
# for s in range(samples):
#     y_tmp = net(x_tmp).detach().numpy()
#     y_samp[s] = y_tmp.reshape(-1)
# plt.plot(x_tmp.numpy(), np.mean(y_samp, axis = 0), label='Mean Posterior Predictive')
# plt.fill_between(x_tmp.numpy().reshape(-1), np.percentile(y_samp, 2.5, axis = 0), np.percentile(y_samp, 97.5, axis = 0), alpha = 0.25, label='95% Confidence')
# plt.legend()
# plt.scatter(x, toy_function(x))
# plt.title('Posterior Predictive')
# plt.show()
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from typing import List


torch.set_float32_matmul_precision('high')

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Attention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.scale = 1.0 / (in_dim ** 0.5)  # 缩放因子

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, v)
        return out


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, time_dim, celltype_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_dim)
        self.celltype_mlp = nn.Linear(celltype_dim, out_dim)
        self.l1 = nn.Linear(in_dim, out_dim)#如果拼接*2
        self.l2 = nn.Linear(out_dim, out_dim)
        self.do1 = nn.Dropout(0.1)#作者是0.1；我改成0看看；还是0.1吧
        self.act = nn.Tanh()
        # self.attention = Attention(out_dim)  # 添加注意力机制;没用
        # Transformer Encoder Layer
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=1)#不行

    def forward(self, x, t, ct):
        h = self.do1(self.act(self.l1(x)))
        time_emb = self.act(self.time_mlp(t)).unsqueeze(1)
        celltype_emb = self.act(self.celltype_mlp(ct)).unsqueeze(1)
        h = h + time_emb + celltype_emb
        # h = self.act(self.l2(h))

        # # 李荣远应用注意力机制
        # h = self.attention(h) + h  # 残差连接

        #lry 加入transform
        # 通过 Transformer 层处理
        # h = h.permute(1, 0, 2)  # 转换为 (seq_len, batch, feature) 形状
        # h = self.transformer_layer(h)  # 应用 Transformer 层
        # h = h.permute(1, 0, 2)  # 转换回 (batch, seq_len, feature)

        h = self.act(self.l2(h))

        return h


    
class GeneEmbeddings(nn.Module):
    def __init__(self, n_gene, gene_dim):
        super().__init__()
        gene_emb = torch.randn(n_gene, gene_dim-1)
        self.gene_emb = nn.Parameter(gene_emb, requires_grad=True)

    def forward(self, x):
        n_cell = x.shape[0]
        batch_gene_emb = self.gene_emb.unsqueeze(0).repeat(n_cell, 1, 1)
        batch_gene_emb = torch.concat([x.unsqueeze(-1), batch_gene_emb], dim=-1)
        return batch_gene_emb
    
class RegDiffusion(nn.Module):
    """
    A RegDiffusion model. For architecture details, please refer to our paper.

    From noise to knowledge: probabilistic diffusion-based neural inference
    
    Args:
        n_genes (int): Number of Genes
        time_dim (int): Dimension of time step embedding
        n_celltype (int): Number of expected cell types. If it is not provided, 
        there would be no celltype embedding. Default is None. 
        celltype_dim (int): Dimension of cell types
        hidden_dims (list[int]): List of integer for the dimensions of the 
        hidden layers. The first hidden dimension will be used as the size
        for gene embedding. 
        adj_dropout (float): A single number between 0 and 1 specifying the 
        percentage of values in the adjacency matrix that are dropped 
        during training. 
        init_coef (int): Coefficient to multiply with gene regulation norm 
        (1/(n_gene - 1)) to initialize the adjacency matrix. 
    """
    def __init__(
        self, n_gene, time_dim, 
        n_celltype=None, celltype_dim=4, 
        hidden_dims=[16, 16, 16], adj_dropout=0.3, init_coef = 5
    ):
        super(RegDiffusion, self).__init__()
        
        self.n_gene = n_gene
        self.gene_dim = hidden_dims[0]
        self.adj_dropout=adj_dropout
        self.gene_reg_norm = 1/(n_gene-1)
        
        adj_A = torch.ones(n_gene, n_gene) * self.gene_reg_norm * init_coef
        self.adj_A = nn.Parameter(adj_A, requires_grad =True, )
        self.sampled_adj_row_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)), 
            requires_grad=False)
        self.sampled_adj_col_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )
        
        self.gene_emb = nn.Sequential(
            GeneEmbeddings(n_gene, self.gene_dim),
            nn.Tanh()
        )
        
        if n_celltype is not None:
            self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)
        
        self.blocks = nn.ModuleList([
            Block(
                hidden_dims[i], hidden_dims[i+1]-1, time_dim, celltype_dim
            ) for i in range(len(hidden_dims) - 1)
        ])


        
        self.final = nn.Linear(hidden_dims[-1]-1, 1)
        
        self.zeros_nonparam = nn.Parameter(
            torch.zeros(n_gene, n_gene), requires_grad=False)
        self.eye_nonparam = nn.Parameter(
            torch.eye(n_gene), requires_grad=False)
        self.mask_nonparam = nn.Parameter(
            1 - torch.eye(n_gene), requires_grad=False)
        
    def soft_thresholding(self, x, tau):
        return torch.sign(x) * torch.max(
            self.zeros_nonparam, torch.abs(x) - tau)
        
    def I_minus_A(self):
        mask = self.mask_nonparam
        if self.train:
            A_dropout = (torch.rand_like(self.adj_A)>self.adj_dropout).float()
            A_dropout /= (1-self.adj_dropout)
            mask = mask * A_dropout
        clean_A = self.soft_thresholding(self.adj_A, self.gene_reg_norm/2)*mask

        return self.eye_nonparam - clean_A
        
    def get_adj_(self):
        return self.soft_thresholding(
            self.adj_A, self.gene_reg_norm/2) * self.mask_nonparam
    
    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()
    
    def get_gene_emb(self):
        return self.gene_emb[0].gene_emb.data.cpu().detach().numpy()
    
    def forward(self, x, t, ct):
        h_time = self.time_mlp(t)
        h_celltype = self.celltype_emb(ct)
        original = x.unsqueeze(-1)
        h_x = self.gene_emb(x)
        for i, block in enumerate(self.blocks):
            if i != 0:
                h_x = torch.concat([h_x, original], dim=-1)
            h_x = block(h_x, h_time, h_celltype)
        
        I_minus_A = self.I_minus_A()
        hz = torch.einsum('ogd,gh->ohd', h_x, I_minus_A)
        z = self.final(hz)
        
        return z.squeeze(-1)


class RegDiffusionPCA(nn.Module):
    """
    A RegDiffusion model. For architecture details, please refer to our paper.

    From noise to knowledge: probabilistic diffusion-based neural inference

    Args:
        n_genes (int): Number of Genes
        time_dim (int): Dimension of time step embedding
        n_celltype (int): Number of expected cell types. If it is not provided,
        there would be no celltype embedding. Default is None.
        celltype_dim (int): Dimension of cell types
        hidden_dims (list[int]): List of integer for the dimensions of the
        hidden layers. The first hidden dimension will be used as the size
        for gene embedding.
        adj_dropout (float): A single number between 0 and 1 specifying the
        percentage of values in the adjacency matrix that are dropped
        during training.
        init_coef (int): Coefficient to multiply with gene regulation norm
        (1/(n_gene - 1)) to initialize the adjacency matrix.
    """

    def __init__(
            self, n_gene, time_dim,
            n_celltype=None, celltype_dim=4,
            hidden_dims=[16, 16, 16], adj_dropout=0.3, init_coef=5
    ):
        super(RegDiffusionPCA, self).__init__()

        self.n_gene = n_gene
        self.gene_dim = hidden_dims[0]
        self.adj_dropout = adj_dropout
        self.gene_reg_norm = 1 / (n_gene - 1)

        adj_A = torch.ones(n_gene, n_gene) * self.gene_reg_norm * init_coef
        self.adj_A = nn.Parameter(adj_A, requires_grad=True, )
        self.sampled_adj_row_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)
        self.sampled_adj_col_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )

        self.gene_emb = nn.Sequential(
            GeneEmbeddings(n_gene, self.gene_dim),
            nn.Tanh()
        )

        if n_celltype is not None:
            self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)

        self.blocks = nn.ModuleList([
            Block(
                hidden_dims[i], hidden_dims[i + 1] - 1, time_dim, celltype_dim
            ) for i in range(len(hidden_dims) - 1)
        ])

        self.final = nn.Linear(hidden_dims[-1] - 1, 1)

        self.zeros_nonparam = nn.Parameter(
            torch.zeros(n_gene, n_gene), requires_grad=False)
        self.eye_nonparam = nn.Parameter(
            torch.eye(n_gene), requires_grad=False)
        self.mask_nonparam = nn.Parameter(
            1 - torch.eye(n_gene), requires_grad=False)

    def soft_thresholding(self, x, tau):
        return torch.sign(x) * torch.max(
            self.zeros_nonparam, torch.abs(x) - tau)

    def I_minus_A(self):
        mask = self.mask_nonparam
        if self.train:
            A_dropout = (torch.rand_like(self.adj_A) > self.adj_dropout).float()
            A_dropout /= (1 - self.adj_dropout)
            mask = mask * A_dropout
        clean_A = self.soft_thresholding(self.adj_A, self.gene_reg_norm / 2) * mask

        return self.eye_nonparam - clean_A

    def get_adj_(self):
        return self.soft_thresholding(
            self.adj_A, self.gene_reg_norm / 2) * self.mask_nonparam

    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()

    def get_gene_emb(self):
        return self.gene_emb[0].gene_emb.data.cpu().detach().numpy()

    def forward(self, x, t, ct):


        h_time = self.time_mlp(t)
        h_celltype = self.celltype_emb(ct)
        original = x.unsqueeze(-1)
        h_x = self.gene_emb(x)
        for i, block in enumerate(self.blocks):
            if i != 0:
                h_x = torch.concat([h_x, original], dim=-1)
            h_x = block(h_x, h_time, h_celltype)

        I_minus_A = self.I_minus_A()
        hz = torch.einsum('ogd,gh->ohd', h_x, I_minus_A)
        z = self.final(hz)

        return z.squeeze(-1)
#来自HyperGVAE
import math
from torch.nn.parameter import Parameter
class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    @staticmethod
    def forward(ctx, M1, M2):

        ctx.save_for_backward(M1, M2)
        return torch.mm(M1.double(), M2.double())

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g.double(), M2.t().double())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t().double(), g.double())

        return g1, g2
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNGenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, nonLinear):
        super(GCNGenerativeNet, self).__init__()

        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(1, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, ):
        x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        output = x_rec

        return output
#lry change add residual,类似残差
class GCNGenerativeNetRe(nn.Module):
    def __init__(self, x_dim, z_dim, nonLinear):
        super(GCNGenerativeNetRe, self).__init__()

        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(2, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, ):
        # x_rec = self.pxz(z.unsqueeze(-1)).squeeze(2)
        x_rec = self.pxz(z).squeeze(2)#输如就是二维
        output = x_rec

        return output
class VGCN(nn.Module):
    def __init__(self, input_dim,  hidden_dim, output_dim, activation=nn.Tanh):
        super(VGCN, self).__init__()
        input_dim = 64  # 因为这里是拼接
        hidden_dim = 128#vae.hidden_dim
        # hidden_dim = 256  # vae.hidden_dim#看看会不会好，也不会
        # activation = torch.nn.Tanh()
        activation = torch.nn.LeakyReLU()#试试会不会好
        # self.labeler = MLP(input_dim, hidden_dim, output_dim, activation)
        # ==============================================#
        # self.gcy1 = GraphConvolution(batchsize, z_dim)
        # self.gcy2 = GraphConvolution(z_dim, n_gene)
        self.gcy=GraphConvolution(input_dim, 64)
        # self.gcy = GraphConvolution(input_dim, 128)#看看会不会好


        self.gcngenerative = GCNGenerativeNet(1, hidden_dim, activation)  #
        # self.gcngenerativeRe = GCNGenerativeNetRe(1, hidden_dim, activation)  #

    def forward(self, x,adj,normal='z-score'):
        # if self.train_on_non_zero:
        #     eval_mask = (x != 0)
        # else:
        #     eval_mask = torch.ones_like(x)
        # adj = torch.tensor(adj.todense()).cuda()
        adj = adj.cuda()
        # x_raw=x.t()
        x_raw=x.clone()
        x_raw=x_raw.t()
        x=x.t()

        if normal == 'noz-score':
            global_mean_cell = x.mean(0)
            global_std_cell = x.std(0)
            x = (x - global_mean_cell) / (global_std_cell)
            # noise = (noise - global_mean) / (global_std)
            x[torch.isnan(x)] = 0
            x[torch.isinf(x)] = 0

        ###############直接最简单的一层线性GCN
        lmean=self.gcy(x,adj)
        lmean=lmean+x#试试加上x
        # ###############假如把自己也拼接上,用这个试试;怎么不行呢，后面在看
        # original=x_raw.unsqueeze(-1)
        # # original = x.unsqueeze(-1)
        # lmean=lmean.unsqueeze(-1)
        # lmean = torch.concat([lmean, original], dim=-1)
        # lmean = lmean.float()
        # gcnout = self.gcngenerativeRe(lmean)  # 换成自编码器不要高斯了，看看
        # ###################
        lmean = lmean.float()
        gcnout = self.gcngenerative(lmean)#换成自编码器不要高斯了，看看


        #######################################################这里加入GenKI的思想，Z*Z不行，用生成的看看
        # adj_rec = torch.matmul(gcnout, gcnout.t())
        #这里对角线值应该设置为0

        loss_rec2 = torch.mean((x - gcnout.float()).pow(2))#如果前面x转置，不用mask，mask=None

        # return gcnout.t(), loss_rec2, adj_rec
        return gcnout.t(), loss_rec2
#lry add rec
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        self.gene_emb = nn.Sequential(
            # GeneEmbeddings(n_gene, self.gene_dim),
            GeneEmbeddings(in_dim, hidden_dim),#lry change
            nn.Tanh()
        )
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.act = activation()
        # self.sigmoid=torch.nn.Sigmoid()
    # def calc_gradient_penalty(self,netD, real_samples, fake_samples):
    #     """Calculates the gradient penalty loss for WGAN GP.
    #        Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
    #        the interpolated real and fake samples, as in the WGAN GP paper.
    #     """
    #     # Random weight term for interpolation between real and fake samples
    #     # alpha = torch.rand(real_samples.size(0), 1,1)#三维，不知道对不对
    #     #     real_samples=real_samples.squeeze(2)
    #     alpha = torch.rand(real_samples.size(0),1)#二维
    #     alpha = alpha.expand(real_samples.size())
    #     #     alpha = alpha.expand(real_samples.size(0), int(real_samples.nelement()/batch_size)).contiguous().view(batch_size, -1)
    #     alpha = alpha.cuda()
    #     #     import pdb
    #     #     pdb.set_trace()
    #     interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    #     # interpolates=interpolates.unsqueeze(2)#这里多一个变3维
    #     # Get random interpolation between real and fake samples
    #     # if use_cuda:
    #     #     interpolates = interpolates.cuda()
    #     interpolates = interpolates.cuda()
    #     interpolates = autograd.Variable(interpolates, requires_grad=True)
    #     disc_interpolates = netD(interpolates)
    #     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
    #                               grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
    #                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    #     gradients = gradients.view(gradients.size(0), -1)
    #     # LAMBDA=10
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    #     return gradient_penalty
    def forward(self, x):
        out1 = self.act(self.l1(x))
        out2 = self.act(self.l2(out1))
        return self.l3(out2)
        # #下面embed 发现没什么大作用，不用了
        # h_x = self.gene_emb(x)#用embeding;输出是【128,910,128】
        # # 将三维矩阵变为二维
        # # 输出形状为 [gene, batchsize * gene_embedding]
        # h_x = h_x.permute(1, 0, 2).reshape(x.shape[1], -1)
        # return h_x
        # return self.sigmoid(self.l3(out2))#注意后面是否是二分类
class MLP_T(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation,time_dim=128,n_celltype=None,celltype_dim=4):
        super(MLP_T, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)#若拼接，乘以2
        self.l2 = nn.Linear(hidden_dim*2+celltype_dim, hidden_dim)#如果下面拼接，就是*2；
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.act = activation()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )
        self.time_mlp2 = nn.Linear(time_dim, out_dim)
        # self.time_mlp2 = nn.Linear(time_dim, out_dim)#lry change
        if n_celltype is not None:
            self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)

    def forward(self, x,t,ct):
        # x=torch.cat((x,x_t),dim=1)
        h_time = self.time_mlp(t)
        h_celltype = self.celltype_emb(ct)
        # time_emb=self.act(self.time_mlp2(h_time)).unsqueeze(1)
        # time_emb = self.act(self.time_mlp2(h_time))
        out1 = self.act(self.l1(x))
        # out1=out1+h_time
        out1=torch.cat((out1,h_time,h_celltype),dim=1)
        # x=x+time_emb
        # out1 = self.act(self.l1(x))
        out2 = self.act(self.l2(out1))
        return self.l3(out2)
class MLP_T1(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation,time_dim=128,n_celltype=None,celltype_dim=4):
        super(MLP_T1, self).__init__()
        self.l1 = nn.Linear(in_dim*2, hidden_dim)#若拼接，乘以2
        self.l2 = nn.Linear(hidden_dim*2+celltype_dim, hidden_dim)#如果下面拼接，就是*2；
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.act = activation()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )
        self.time_mlp2 = nn.Linear(time_dim, out_dim)
        # self.time_mlp2 = nn.Linear(time_dim, out_dim)#lry change
        if n_celltype is not None:
            self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)

    def forward(self, x,t,x_t,ct):
        x=torch.cat((x,x_t),dim=1)
        h_time = self.time_mlp(t)
        h_celltype = self.celltype_emb(ct)
        # time_emb=self.act(self.time_mlp2(h_time)).unsqueeze(1)
        # time_emb = self.act(self.time_mlp2(h_time))
        out1 = self.act(self.l1(x))
        # out1=out1+h_time
        out1=torch.cat((out1,h_time,h_celltype),dim=1)
        # x=x+time_emb
        # out1 = self.act(self.l1(x))
        out2 = self.act(self.l2(out1))
        return self.l3(out2)
class RegDiffusionRec(nn.Module):
    """
    A RegDiffusionRec model. For architecture details, please refer to our paper.

    From noise to knowledge: probabilistic diffusion-based neural inference

    Args:
        n_genes (int): Number of Genes
        time_dim (int): Dimension of time step embedding
        n_celltype (int): Number of expected cell types. If it is not provided,
        there would be no celltype embedding. Default is None.
        celltype_dim (int): Dimension of cell types
        hidden_dims (list[int]): List of integer for the dimensions of the
        hidden layers. The first hidden dimension will be used as the size
        for gene embedding.
        adj_dropout (float): A single number between 0 and 1 specifying the
        percentage of values in the adjacency matrix that are dropped
        during training.
        init_coef (int): Coefficient to multiply with gene regulation norm
        (1/(n_gene - 1)) to initialize the adjacency matrix.
    """

    def __init__(
            self, n_gene, time_dim,
            n_celltype=None, celltype_dim=4,
            hidden_dims=[16, 16, 16], adj_dropout=0.3, init_coef=5
    ):
        super(RegDiffusionRec, self).__init__()

        self.n_gene = n_gene
        self.gene_dim = hidden_dims[0]
        self.adj_dropout = adj_dropout
        self.gene_reg_norm = 1 / (n_gene - 1)

        adj_A = torch.ones(n_gene, n_gene) * self.gene_reg_norm * init_coef
        self.adj_A = nn.Parameter(adj_A, requires_grad=True, )
        self.sampled_adj_row_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)
        self.sampled_adj_col_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )

        self.gene_emb = nn.Sequential(
            GeneEmbeddings(n_gene, self.gene_dim),
            nn.Tanh()
        )

        if n_celltype is not None:
            self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)

        self.blocks = nn.ModuleList([
            Block(
                hidden_dims[i], hidden_dims[i + 1] - 1, time_dim, celltype_dim
            ) for i in range(len(hidden_dims) - 1)
        ])

        self.final = nn.Linear(hidden_dims[-1] - 1, 1)

        #lry add Rec---------------------------------------------------
        self.generative_pxz = MLP(1, hidden_dims[1], 1, nn.Tanh)  # share Decoder,n_gene
        # lry add Rec---------------------------------------------------

        self.zeros_nonparam = nn.Parameter(
            torch.zeros(n_gene, n_gene), requires_grad=False)
        self.eye_nonparam = nn.Parameter(
            torch.eye(n_gene), requires_grad=False)
        self.mask_nonparam = nn.Parameter(
            1 - torch.eye(n_gene), requires_grad=False)

    def soft_thresholding(self, x, tau):
        return torch.sign(x) * torch.max(
            self.zeros_nonparam, torch.abs(x) - tau)

    def I_minus_A(self):
        mask = self.mask_nonparam
        if self.train:
            A_dropout = (torch.rand_like(self.adj_A) > self.adj_dropout).float()
            A_dropout /= (1 - self.adj_dropout)
            mask = mask * A_dropout
        clean_A = self.soft_thresholding(self.adj_A, self.gene_reg_norm / 2) * mask

        return self.eye_nonparam - clean_A

    def get_adj_(self):
        return self.soft_thresholding(
            self.adj_A, self.gene_reg_norm / 2) * self.mask_nonparam

    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()

    def get_gene_emb(self):
        return self.gene_emb[0].gene_emb.data.cpu().detach().numpy()

    def forward(self, x, t, ct):
        h_time = self.time_mlp(t)
        h_celltype = self.celltype_emb(ct)
        original = x.unsqueeze(-1)
        h_x = self.gene_emb(x)
        for i, block in enumerate(self.blocks):
            if i != 0:
                h_x = torch.concat([h_x, original], dim=-1)
            h_x = block(h_x, h_time, h_celltype)

        I_minus_A = self.I_minus_A()
        hz = torch.einsum('ogd,gh->ohd', h_x, I_minus_A)
        z = self.final(hz)
        #lry add rec-------------------------------------
        x_rec=self.generative_pxz(z)

        # return z.squeeze(-1)
        return z.squeeze(-1),x_rec.squeeze(-1)
#lry add wgan
from torch import autograd
class RegDiffusionDWGAN(nn.Module):
    """
    A RegDiffusionRec model. For architecture details, please refer to our paper.

    From noise to knowledge: probabilistic diffusion-based neural inference

    Args:
        n_genes (int): Number of Genes
        time_dim (int): Dimension of time step embedding
        n_celltype (int): Number of expected cell types. If it is not provided,
        there would be no celltype embedding. Default is None.
        celltype_dim (int): Dimension of cell types
        hidden_dims (list[int]): List of integer for the dimensions of the
        hidden layers. The first hidden dimension will be used as the size
        for gene embedding.
        adj_dropout (float): A single number between 0 and 1 specifying the
        percentage of values in the adjacency matrix that are dropped
        during training.
        init_coef (int): Coefficient to multiply with gene regulation norm
        (1/(n_gene - 1)) to initialize the adjacency matrix.
    """

    def __init__(
            self, n_gene, time_dim,
            n_celltype=None, celltype_dim=4,
            hidden_dims=[16, 16, 16], adj_dropout=0.3, init_coef=5
    ):
        super(RegDiffusionDWGAN, self).__init__()

        self.n_gene = n_gene
        self.gene_dim = hidden_dims[0]
        self.adj_dropout = adj_dropout
        self.gene_reg_norm = 1 / (n_gene - 1)

        adj_A = torch.ones(n_gene, n_gene) * self.gene_reg_norm * init_coef
        self.adj_A = nn.Parameter(adj_A, requires_grad=True, )
        self.sampled_adj_row_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)
        self.sampled_adj_col_nonparam = nn.Parameter(
            torch.randint(0, n_gene, size=(10000,)),
            requires_grad=False)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Tanh()
        )

        self.gene_emb = nn.Sequential(
            GeneEmbeddings(n_gene, self.gene_dim),
            nn.Tanh()
        )
        #如果前面拼接

        if n_celltype is not None:
            self.celltype_emb = nn.Embedding(n_celltype, celltype_dim)

        self.blocks = nn.ModuleList([
            Block(
                hidden_dims[i], hidden_dims[i + 1] - 1, time_dim, celltype_dim
            ) for i in range(len(hidden_dims) - 1)
        ])


        #lry add transformer-----------------------------------
        # Transformer Encoder Setup
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dims[-1]-1, nhead=1)# nhead=n_heads=8
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)#n_layers

        #lry add transformer end -----------------------------
        self.final = nn.Linear(hidden_dims[-1] - 1, 1)

        # self.first = nn.Linear(256, 128)#如果拼接多加这一层神经网络;batch_size*2;n_gene。需要转置

        #lry add Rec---------------------------------------------------
        # self.discriminator = MLP(n_gene, hidden_dim, n_gene, activation)  # 感觉最后输出你只是判断真假，1就行，不用n_gene
        # self.discriminator = MLP(1, 128, 1, nn.LeakyReLU)  # nn.Ta;共享
        # self.discriminator = MLP(n_gene, 128, n_gene, nn.LeakyReLU)
        self.vgcn_model = VGCN(64, 128, 1, torch.nn.Tanh())#batchsize  ,hidden,outdim=1但是没用到
        # self.discriminator = MLP(1, 256, 1, nn.LeakyReLU)  # nn.Ta
        # self.discriminator = MLP_T(n_gene, 128, n_gene, nn.Tanh, 128, n_celltype, celltype_dim)  # nn.Ta
        # self.discriminator = MLP(128, 1024, 128, nn.Tanh)  # #假如改为从基因层面来，也就是转置处理
        # self.discriminator_t = MLP_T(n_gene, 128, n_gene, nn.Tanh,128,n_celltype,celltype_dim)  # nn.Ta
        # self.final_mlp= MLP(n_gene, 128, n_gene, nn.Tanh)

        # self.generative_pxz = MLP(1, hidden_dims[1], 1, nn.Tanh)  # share Decoder,n_gene
        # lry add Rec---------------------------------------------------


        self.zeros_nonparam = nn.Parameter(
            torch.zeros(n_gene, n_gene), requires_grad=False)
        self.eye_nonparam = nn.Parameter(
            torch.eye(n_gene), requires_grad=False)
        self.mask_nonparam = nn.Parameter(
            1 - torch.eye(n_gene), requires_grad=False)

    # def soft_thresholding(self, x, tau):
    #     return torch.sign(x) * torch.max(
    #             self.zeros_nonparam, torch.abs(x) - tau)
    #lry change
    def soft_thresholding(self, x, tau):

        #改1;String好，non差
        # alpha=0.5
        # l1_part = torch.sign(x) * torch.relu(torch.abs(x) - tau)
        # l2_part = x / (1 + 2 * (1 - alpha) * tau)
        # return alpha * l1_part + (1 - alpha) * l2_part
        # 改2 String好，non差
        # alpha = 10
        # abs_x = torch.abs(x)
        # # 计算平滑过渡因子
        # smooth_factor = 0.5 * (1 + torch.tanh(alpha * (abs_x - tau)))
        # # 平滑阈值输出
        # return torch.sign(x) * (abs_x - tau * smooth_factor) * (abs_x > 0).float()
        #改3；不太行
        # lambda_val=tau
        # mix_ratio = 0.5#mix_ratio: 软阈值输出占比（0:纯硬阈值, 1:纯软阈值）
        # abs_x = torch.abs(x)
        # # 软阈值部分
        # soft_part = torch.sign(x) * torch.relu(abs_x - lambda_val)
        # # 硬阈值部分（完全保留或置零）
        # hard_part = x * (abs_x > lambda_val).float()
        # # 混合输出
        # return mix_ratio * soft_part + (1 -mix_ratio) * hard_part
        # #改4
        # lambda_k=5
        # output=torch.sign(x)*torch.where(torch.abs(x)>tau,torch.abs(x)-tau,(torch.abs(x)-tau)*torch.exp(-lambda_k*(tau-torch.abs(x))))
        # # 改5
        # lambda_k = 1 / tau
        output=torch.sign(x)*torch.where(torch.abs(x)>tau,torch.abs(x)-tau,(torch.abs(x)-tau)*torch.exp(-(tau-torch.abs(x))))
        return output

        # lambda_val=tau
        # # return torch.sign(x) * torch.relu(torch.abs(x) - lambda_val)#同原来的作者
        # return torch.sign(x) * torch.exp(torch.abs(x) - lambda_val) *0.001 # 同原来的作者
        # #下面这个也不行
        # alpha = 100
        # abs_x = torch.abs(x)
        # # 平滑过渡区域 (|x| ∈ [λ-ε, λ+ε])
        # mask = (abs_x > lambda_val - 1 / alpha) & (abs_x < lambda_val + 1 / alpha)
        # # 硬阈值部分
        # linear_part = torch.sign(x) * (abs_x - lambda_val)
        # # 平滑过渡部分
        # smooth_part = torch.sign(x) * 0.5 * alpha * (abs_x - (lambda_val - 1 / alpha)) ** 2
        # return torch.where(mask, smooth_part, linear_part) * (abs_x > lambda_val).float()

        # return torch.sign(x) * torch.exp(torch.abs(x) - tau)#不行
        # return torch.exp(torch.abs(x) - tau)#不行
        # return torch.sign(x) * torch.where(x>tau,x*1.5,0)  # 不行
        # adjusted_A=torch.where(torch.abs(x) > tau, x * 1.1, x)  # 不行
        # adjusted_A = torch.where(adjusted_A < tau, tau, adjusted_A)
        # return adjusted_A
        # return torch.sign(x) * torch.max(
        #         self.zeros_nonparam, torch.abs(x) - tau)
        # return  torch.sign(x)*torch.where(torch.abs(x) > tau, x , self.zeros_nonparam)#不太行
        # return torch.where(torch.abs(x) > tau, x * 1.1, self.zeros_nonparam)  #
        # return torch.sign(x) * (torch.abs(x) - tau)#这种方式String可以，但Non不行
        # return torch.sign(x) * 0.001*torch.exp(torch.abs(x) - tau)
        # return  torch.exp(torch.abs(x))#不行

    def I_minus_A(self):
        mask = self.mask_nonparam
        if self.train:
            A_dropout = (torch.rand_like(self.adj_A) > self.adj_dropout).float()
            A_dropout /= (1 - self.adj_dropout)
            mask = mask * A_dropout
        clean_A = self.soft_thresholding(self.adj_A, self.gene_reg_norm / 2) * mask

        return self.eye_nonparam - (clean_A)

    def get_adj_(self):
        return self.soft_thresholding(
            self.adj_A, self.gene_reg_norm / 2) * self.mask_nonparam

    def get_adj(self):
        adj = self.get_adj_().detach().cpu().numpy() / self.gene_reg_norm
        return adj.astype(np.float16)
    #-------------lry add
    def calc_gradient_penalty(self,netD, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP.
           Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
           the interpolated real and fake samples, as in the WGAN GP paper.
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1)
        #     real_samples=real_samples.squeeze(2)
        #     alpha = torch.rand(real_samples.size(0),1)
        alpha = alpha.expand(real_samples.size())
        #     alpha = alpha.expand(real_samples.size(0), int(real_samples.nelement()/batch_size)).contiguous().view(batch_size, -1)
        alpha = alpha.cuda()
        #     import pdb
        #     pdb.set_trace()
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        # interpolates=interpolates.unsqueeze(2)#这里多一个变3维
        # Get random interpolation between real and fake samples
        # if use_cuda:
        #     interpolates = interpolates.cuda()
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        # LAMBDA=10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def calc_gradient_penalty_t(self,netD, real_samples, fake_samples,t,ct):
        """Calculates the gradient penalty loss for WGAN GP.
           Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
           the interpolated real and fake samples, as in the WGAN GP paper.
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1)
        #     real_samples=real_samples.squeeze(2)
        #     alpha = torch.rand(real_samples.size(0),1)
        alpha = alpha.expand(real_samples.size())
        #     alpha = alpha.expand(real_samples.size(0), int(real_samples.nelement()/batch_size)).contiguous().view(batch_size, -1)
        alpha = alpha.cuda()
        #     import pdb
        #     pdb.set_trace()
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        # interpolates=interpolates.unsqueeze(2)#这里多一个变3维
        # Get random interpolation between real and fake samples
        # if use_cuda:
        #     interpolates = interpolates.cuda()
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates,t,ct)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        # LAMBDA=10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty
    #------------------------------lry add
    def calc_gradient_penalty_t1(self,netD, real_samples, fake_samples,t,x_tp1,ct):
        """Calculates the gradient penalty loss for WGAN GP.
           Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
           the interpolated real and fake samples, as in the WGAN GP paper.
        """
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1)
        #     real_samples=real_samples.squeeze(2)
        #     alpha = torch.rand(real_samples.size(0),1)
        alpha = alpha.expand(real_samples.size())
        #     alpha = alpha.expand(real_samples.size(0), int(real_samples.nelement()/batch_size)).contiguous().view(batch_size, -1)
        alpha = alpha.cuda()
        #     import pdb
        #     pdb.set_trace()
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        # interpolates=interpolates.unsqueeze(2)#这里多一个变3维
        # Get random interpolation between real and fake samples
        # if use_cuda:
        #     interpolates = interpolates.cuda()
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates,t,x_tp1,ct)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        # LAMBDA=10
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty
    #------------------------------lry add

    @torch.no_grad()
    def get_sampled_adj_(self):
        return self.get_adj_()[
            self.sampled_adj_row_nonparam, self.sampled_adj_col_nonparam
        ].detach()

    def get_gene_emb(self):
        return self.gene_emb[0].gene_emb.data.cpu().detach().numpy()

    def forward(self, x, t, ct):
        # x0_fun=self.discriminator
        x0_fun = self.vgcn_model
        h_time = self.time_mlp(t)
        h_celltype = self.celltype_emb(ct)
        # x=self.first(x.T).T#如果拼接多加一个神经网络
        original = x.unsqueeze(-1)
        h_x = self.gene_emb(x)
        for i, block in enumerate(self.blocks):
            if i != 0:
                h_x = torch.concat([h_x, original], dim=-1)
            h_x = block(h_x, h_time, h_celltype)

        # # Transformer Encoding
        # h_x = self.transformer_encoder(h_x.permute(1, 0, 2))  # 转换为 (seq_len, batch, feature) 形状
        # h_x = h_x.permute(1, 0, 2)  # 转换回 (batch, seq_len, feature)

        I_minus_A = self.I_minus_A()
        hz = torch.einsum('ogd,gh->ohd', h_x, I_minus_A)
        # hz = torch.einsum('ogd,gh->ohd', h_x, self.adj_A)#lry change: self.adj_A
        z = self.final(hz)
        # z2 = torch.einsum('ogd,gh->ohd', z, I_minus_A)  # lry change: self.adj_A
        # z = torch.einsum('ogd,gh->ohd', z, torch.inverse(I_minus_A))

        # # # #lry add rec-------------------------------------
        # z_inv = torch.einsum('ogd,gh->ohd', z ,torch.inverse(I_minus_A) )
        # x_t = self.generative_pxz(z_inv)
        # z=self.generative_pxz(z)


        # return z.squeeze(-1),x_t.squeeze(-1)#lry change
        return z.squeeze(-1),x0_fun
        # return z.squeeze(-1)
        # return z.squeeze(-1),h_x.squeeze(-1)#lry change
        # return  x_t.squeeze(-1)  # lry change
        # return z.squeeze(-1),x_rec.squeeze(-1)
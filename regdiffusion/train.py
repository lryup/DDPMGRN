import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from .models import RegDiffusion,RegDiffusionRec,RegDiffusionPCA,RegDiffusionDWGAN,MLP_T,MLP
from tqdm import tqdm
from .logger import LightLogger
from datetime import datetime
from .grn import GRN
from .evaluator import GRNEvaluator
from .logger import LightLogger
import matplotlib.pyplot as plt
import warnings

#copy   stDCL
def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
def linear_beta_schedule(timesteps, start_noise, end_noise):
    scale = 1000 / timesteps
    beta_start = scale * start_noise
    beta_end = scale * end_noise
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float)

def power_beta_schedule(timesteps, start_noise, end_noise, power=2):
    linspace = torch.linspace(0, 1, timesteps, dtype = torch.float)
    poweredspace = linspace ** power
    scale = 1000 / timesteps
    beta_start = scale * start_noise
    beta_end = scale * end_noise
    return beta_start + (beta_end - beta_start) * poweredspace



#
def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out
#
class DDPMGRN:
    """
    Initialize and Train a RegDiffusion model.

    For architecture and training details, please refer to our paper.

    > From noise to knowledge: probabilistic diffusion-based neural inference

    You can access the model through `RegDiffusionTrainerRec.model`.

    Args:
        exp_array (np.ndarray): 2D numpy array. If used on single-cell RNAseq,
            the rows are cells and the columns are genes. Data should be log
            transformed. You may also want to remove all non expressed genes.
        cell_types (np.ndarray): (Optional) 1D integer array for cell type. If
            you have labels in your cell type, you need to convert them to
            interge. Default is None.
        T (int): Total number of diffusion steps. Default: 5,000
        start_noise (float): Minimal noise level (beta) to be added. Default:
            0.0001
        end_noise (float): Maximal noise level (beta) to be added. Default:
            0.02
        time_dim (int): Dimension size for the time embedding. Default: 64.
        celltype_dim (int): Dimension size for the cell type embedding.
            Default: 4.
        hidden_dim (list): Dimension sizes for the feature learning layers. We
            use the size of the first layer as the dimension for gene embeddings
            as well. Default: [16, 16, 16].
        init_coef (int): A coefficent to control the value to initialize the
            adjacency matrix. Here we define regulatory norm as 1 over (number
            of genes - 1). The value which we use to initialize the model is
            `init_coef` times of the regulatory norm. Default: 5.
        lr_nn (float): Learning rate for the rest of the neural networks except
            the adjacency matrix. Default: 0.001
        lr_adj (float): Learning rate for the adjacency matrix. By default, it
            equals to 0.02 * gene regulatory norm, which equals 1/(n_gene-1).
        weight_decay_nn (float): L2 regularization coef on the rest of the
            neural networks. Default: 0.1.
        weight_decay_adj (float): L2 regularization coef on the adj matrix.
            Default: 0.01.
        sparse_loss_coef (float): L1 regularization coef on the adj matrix.
            Default: 0.25.
        adj_dropout (float): Probability of an edge to be zeroed. Default: 0.3.
        batch_size (int): Batch size for training. Default: 128.
        n_steps (int): Total number of training iterations. Default: 1000.
        train_split (float): Train partition. Default: 1.0.
        train_split_seed (int): Random seed for train/val partition.
            Default: 123
        device (str or torch.device): Device where the model is running. For
            example, "cpu", "cuda", "cuda:1", and etc. You are not recommended
            to run this model on Apple's MPS chips. Default is "cuda" but if
            you only has CPU, it will switch back to CPU.
        compile (boolean): Whether to compile the model before training.
            Compile the model is a good idea on large dataset and ofter improves
            inference speed when it works. For smaller dataset, eager execution
            if often good enough.
        evaluator (GRNEvaluator): (Optional) A defined GRNEvaluator if ground
            truth data is available. Evaluation will be done every 100 steps by
            default but you can change this setting through the eval_on_n_steps
            option. Default is None
        eval_on_n_steps (int): If an evaluator is provided, the trainer will
            run evaluation every `eval_on_n_steps` steps. Default: 100.
        logger (LightLogger): (Optional) A LightLogger to log training process.
            The only situation when you need to provide this is when you want
            to save logs from different trainers into the same logger. Default
            is None.
    """


    def __init__(
            self, exp_array, cell_types=None,
            T=5000, start_noise=0.0001, end_noise=0.02,
            time_dim=64, celltype_dim=4, hidden_dims=[16, 16, 16],
            init_coef=5,
            lr_nn=1e-3, lr_adj=None,
            weight_decay_nn=0.1, weight_decay_adj=0.01,
            sparse_loss_coef=0.25, adj_dropout=0.30,
            batch_size=128, n_steps=1000,
            train_split=1.0, train_split_seed=123,
            device='cuda', compile=False,
            evaluator=None, eval_on_n_steps=100, logger=None

    ):
        hp = locals()
        del hp['exp_array']
        del hp['cell_types']
        del hp['logger']
        self.hp = hp

        if device == 'mps':
            raise Exception("We noticed unreliable training behavior on",
                            "Apple's silicon. Consider using other devices.")
        elif device.startswith('cuda'):
            if not torch.cuda.is_available():
                print(
                    "You specified cuda as your computing device but apprently",
                    "it's not available. Setting device to cpu for now. ")
                device = 'cpu'
        self.device = device
        self.hp['device'] = device

        # Logger ---------------------------------------------------------------
        if logger is None:
            self.logger = LightLogger()
        self.note_id = self.logger.start()

        # Define diffusion schedule
        self.betas = linear_beta_schedule(T, start_noise, end_noise).to(device)
        self.alphas = 1. - self.betas
        alpha_bars = torch.cumprod(self.alphas, axis=0)
        self.mean_schedule = torch.sqrt(alpha_bars).to(device)
        self.std_schedule = torch.sqrt(1. - alpha_bars).to(device)

        #lry add ,删除  一  bar
        self.mean_schedule_nobar = torch.sqrt(self.alphas).to(device)
        self.std_schedule_nobar = torch.sqrt(1. - self.alphas).to(device)

        # self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        # self.a_s_cum = np.cumprod(self.a_s.cpu())
        # self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        # self.a_s_prev = self.a_s.clone()
        # self.a_s_prev[-1] = 1
        #
        # self.a_s_cum = self.a_s_cum.to(device)
        # self.sigmas_cum = self.sigmas_cum.to(device)
        # self.a_s_prev = self.a_s_prev.to(device)


        self.alphas_cumprod = alpha_bars.to(device)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        ).to(device)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod).to(device)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)).to(device)
        self.posterior_mean_coef2 = (
                    (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)).to(device)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20)).to(device)

        self.posterior_mean_coef1_lry=1/torch.sqrt(self.alphas)
        self.posterior_mean_coef2_lry=-(self.posterior_mean_coef1_lry)*((1-self.alphas)/torch.sqrt(1-self.alphas_cumprod))

        # Prepare Data ---------------------------------------------------------
        if (exp_array.sum(0) == 0).sum() > 0:
            warnings.warn(
                "Some columns in the exp_array contains all zero values, "
                "which often causes trouble in inference. Please consider "
                "removing these columns before continuing. "
            )
        if cell_types is None:
            cell_types = np.zeros(exp_array.shape[0], dtype=int)
        self.n_celltype = len(np.unique(cell_types))
        n_cell, n_gene = exp_array.shape
        self.n_cell = n_cell
        self.n_gene = n_gene

        self.evaluator = evaluator


        ## Normalize data
        cell_min = exp_array.min(axis=1, keepdims=True)
        cell_max = exp_array.max(axis=1, keepdims=True)
        normalized_X = (exp_array - cell_min) / (cell_max - cell_min)
        normalized_X = (normalized_X - normalized_X.mean(0)) / normalized_X.std(0)



        ## Train/validation split
        random_state = np.random.RandomState(train_split_seed)
        train_val_split = random_state.rand(normalized_X.shape[0])
        train_index = train_val_split <= train_split
        val_index = train_val_split > train_split

        x_tensor_train = torch.tensor(
            normalized_X[train_index,], dtype=torch.float32)
        celltype_tensor_train = torch.tensor(
            cell_types[train_index], dtype=int)
        x_tensor_val = torch.tensor(
            normalized_X[val_index,], dtype=torch.float32)
        celltype_tensor_val = torch.tensor(cell_types[val_index], dtype=int)


        ## Setup dataset and dataloader
        self.train_dataset = torch.utils.data.TensorDataset(
            x_tensor_train, celltype_tensor_train
        )


        # Implement bootstrap for train sampler
        train_sampler = torch.utils.data.RandomSampler(
            self.train_dataset, replacement=True, num_samples=batch_size)
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            drop_last=True)

        self.val_dataset = torch.utils.data.TensorDataset(
            x_tensor_val, celltype_tensor_val
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False)

        # Setup Model ----------------------------------------------------------
        gene_reg_norm = 1 / (n_gene - 1)
        self.model = RegDiffusionDWGAN(
            n_gene=n_gene,
            time_dim=time_dim,
            n_celltype=self.n_celltype,
            celltype_dim=celltype_dim,
            hidden_dims=hidden_dims,
            adj_dropout=adj_dropout,
            init_coef=init_coef
        )


        # Setup optimizer ------------------------------------------------------
        if lr_adj is None:
            lr_adj = gene_reg_norm / 50
            self.hp['lr_adj'] = lr_adj
        adj_params = []
        non_adj_params = []
        for name, param in self.model.named_parameters():
            if name.endswith('adj_A'):
                adj_params.append(param)
            else:
                if not name.endswith('_nonparam'):
                    non_adj_params.append(param)
        self.opt = torch.optim.Adam(
            [{'params': non_adj_params}, {'params': adj_params}],
            lr=lr_nn,
            weight_decay=weight_decay_nn, betas=[0.9, 0.99]
        )
        self.opt.param_groups[0]['lr'] = lr_nn
        self.opt.param_groups[1]['lr'] = lr_adj
        self.opt.param_groups[1]['weight_decay'] = weight_decay_adj


        self.model.to(device)
        if self.device.startswith('cuda') and compile:
            self.original_model = self.model
            self.model = torch.compile(self.model)
        self.total_time_cost = 0
        self.losses_on_gene = None
        self.model_name = 'RegDiffusion'

    @torch.no_grad()
    def forward_pass(self, x_0, t):
        """
        Forward diffusion process

        Args:
            x_0 (torch.FloatTensor): Torch tensor for expression data. Rows are
            cells and columns are genes
            t (torch.LongTensor): Torch tensor for diffusion time steps.
        """
        noise = torch.randn_like(x_0)
        mean_coef = self.mean_schedule.gather(dim=-1, index=t)
        std_coef = self.std_schedule.gather(dim=-1, index=t)
        x_t = mean_coef.unsqueeze(-1) * x_0 + std_coef.unsqueeze(-1) * noise
        return x_t, noise

    # 计算基因之间的余弦相似性
    def compute_cosine_similarity(self,gene_matrix):
        # 计算基因的 L2 范数
        norms = gene_matrix.norm(dim=0, keepdim=True)  # 对每一列（基因）计算范数
        normalized_matrix = gene_matrix / norms  # 归一化矩阵
        similarity_matrix = torch.mm(normalized_matrix.T, normalized_matrix)  # 计算相似性矩阵
        return similarity_matrix

    # 计算基因之间的欧几里得距离
    def compute_euclidean_distance(self,gene_matrix):
        # 使用广播计算距离矩阵
        distances = torch.cdist(gene_matrix.T, gene_matrix.T, p=2)  # 转置以计算基因之间的距离
        return distances


    def train(self, n_steps=None,bl_gt=None, bl_dt_var_names=None,Lsim=None):#修改加入 bl_gt=None, bl_dt_var_names=None#
        """
        Train the initialized model for a number of steps.

        Args:
            n_steps (int): Number of steps to train. If not provided, it will
                train the model by the n_steps sepcified in class
                initialization. Please read our paper to see how to identify
                the converge point.
        """
        start_time = datetime.now()
        eval_steps = self.hp['eval_on_n_steps']
        if n_steps is None:
            n_steps = self.hp['n_steps']
        sampled_adj = self.model.get_sampled_adj_()

        with tqdm(range(n_steps)) as pbar:
            # if bl_gt is not None:
            for epoch in pbar:#
            # for epoch in range(n_steps):  # Removed tqdm
                epoch_loss = []
                for step, batch in enumerate(self.train_dataloader):
                    x_0, ct = batch
                    x_0 = x_0.to(self.device)
                    ct = ct.to(self.device)
                    self.opt.zero_grad()
                    t = torch.randint(
                        0, self.hp['T'], (x_0.shape[0],),
                        device=self.device
                    ).long()

                    x_noisy, noise = self.forward_pass(x_0, t)
                    # z = self.model(x_noisy, t, ct)
                    z,x0_fun = self.model(x_noisy, t, ct)


                    # 参考谢赛宁
                    if True:
                        real = x_noisy.T
                        # real = x_tp1.T
                        norm_A = torch.norm(real, dim=1, keepdim=True)
                        # 归一化张量
                        A_normalized = real / norm_A
                        # A_normalized = D_real #不归一
                        # errD_real = torch.mm(A_normalized, A_normalized.t())
                        fake = z.T
                        norm_A = torch.norm(fake, dim=1, keepdim=True)
                        # 归一化张量
                        B_normalized = fake / norm_A
                        Sim = torch.mm(A_normalized, B_normalized.T)
                        # 将对角线元素乘以 -1
                        diag_indices = torch.arange(Sim.size(0))  # 对角线的索引
                        Sim[diag_indices, diag_indices] *= -1  # 将对角线元素乘以 -1

                        # loss2 = torch.abs(loss2)
                        # 取指数
                        # loss2 = torch.exp(-loss2)
                        loss2 = Sim.mean()
                        # 这里只是后面保存看损失，真实训练不需要
                        if False:
                            # 计算对角元素的均值
                            diag_mean = Sim.diag().mean()
                            # 计算非对角元素的均值
                            # 创建一个布尔掩码，选取非对角元素
                            mask = torch.ones_like(Sim, dtype=torch.bool)
                            mask[diag_indices, diag_indices] = False  # 将对角线位置设为 False
                            # 计算非对角元素的均值
                            off_diag_mean = Sim[mask].mean()



                    import torch.nn.functional as F
                    loss_ = F.mse_loss(noise, z, reduction='none')  # l2

                    loss = loss_.mean()


                    loss += Lsim* loss2#

                    adj_m = self.model.get_adj_()
                    loss_sparse = adj_m.mean() * self.hp['sparse_loss_coef']

                    if epoch > 10:
                        loss = loss + loss_sparse
                    loss.backward()
                    self.opt.step()
                    epoch_loss.append(loss.item())
                train_loss = np.mean(epoch_loss)
                sampled_adj_new = self.model.get_sampled_adj_()
                adj_diff = (
                                   sampled_adj_new - sampled_adj
                           ).mean().item() * (self.n_gene - 1)
                sampled_adj = sampled_adj_new
                # pbar.set_description(
                #     f' loss: {train_loss:.5f}, Change on Adj: {adj_diff:.5f},errG: {errG:.5f},'
                #     f'errD: {errD:.5f},recD: {recD:.5f}')
                #这个是显示进度条
                # pbar.set_description(
                #     f' loss: {train_loss:.5f}, Adj: {adj_diff:.5f},'
                # )
                # print( 'loss:', loss)
                #epoch_log = {'train_loss': train_loss, 'adj_change': adj_diff}


                #改成下列的
                if bl_gt is not None:
                    from regdiffusion import evaluator
                    evaluator = evaluator.GRNEvaluator(bl_gt, bl_dt_var_names)
                    ppi_auc = evaluator.evaluate(
                        self.model.get_adj()
                    )
                    ppi_auc = {key: round(value, 5) for key, value in ppi_auc.items()}#保留5位小数
                else:
                    ppi_auc=0
                # print(ppi_auc)
                epoch_log = {
                    'epoch:':format(epoch),
                    'train_loss': format(train_loss, '.5f'),
                    'loss_mse': format(loss_.mean(), '.5f'),
                    'loss_cos': format(loss2, '.5f'),
                    'loss_sparse': format(loss_sparse, '.5f'),
                    'adj_change':format(adj_diff, '.5f'),
                    #'diag_mean':format(diag_mean,'.5f'),#看损失
                    #'off_diag_mean ': format(off_diag_mean , '.5f'),
                    'ppi_auc':ppi_auc


                }
                # print(epoch_log)#是否保存csv看损失情况
                #_______________________________
                if epoch % eval_steps == eval_steps - 1:
                    if self.evaluator is not None:
                        eval_result = self.evaluator.evaluate(
                            self.model.get_adj()
                        )
                        # print(eval_result)
                        for k in eval_result.keys():
                            epoch_log[k] = eval_result[k]
                    if self.hp['train_split'] < 1:
                        with torch.no_grad():
                            val_epoch_loss = []
                            for step, batch in enumerate(self.val_dataloader):
                                x_0, ct = batch
                                x_0 = x_0.to(self.device)
                                ct = ct.to(self.device)
                                t = torch.randint(
                                    0, self.hp['T'], (x_0.shape[0],),
                                    device=self.device).long()

                                x_noisy, noise = self.forward_pass(x_0, t)
                                z = self.model(x_noisy, t, ct)
                                step_val_loss = F.mse_loss(
                                    noise, z, reduction='mean').item()
                                val_epoch_loss.append(step_val_loss)
                            epoch_log['val_loss'] = np.mean(val_epoch_loss)
                self.logger.log(epoch_log)
                if bl_gt is not None:
                    print(epoch_log)
        self.losses_on_gene = loss_.detach().mean(0).cpu().numpy()
        self.total_time_cost += int(
            (datetime.now() - start_time).total_seconds())


    def training_curves(self):
        """
        Plot out the training curves on `train_loss` and `adj_change`. Check
        out our paper for how to use `adj_change` to identify the convergence
        point.
        """
        log_df = self.logger.to_df()
        if 'train_loss' in log_df:
            figure, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].plot(log_df['train_loss'])
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Training Loss')
            axes[1].plot(log_df['adj_change'][1:])
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Amount of Change in Adj. Matrix')
            plt.show()
        else:
            print(
                'Training log and Adj Change are not available. Train your ',
                'model using the .train() method.')

    def get_grn(self, gene_names, tf_names=None, top_gene_percentile=None):
        """
        Obtain a GRN object. You need to provide the genes names.

        Args:
            gene_names (np.ndarray): An array of names of all genes. The order
                of genes should be the same as the order used in your expression
                table.
            tf_names (np.ndarray):An array of names of all transcriptional
                factors. The order of genes should be the same as the order
                used in your expression table.
            top_gene_percentile (int): If provided, we will set the value on
                weak links to be zero. It is useful if you want to save the
                regulatory relationship in a GRN object as a sparse matrix.

        """
        adj = self.model.get_adj()
        return GRN(adj, gene_names, tf_names, top_gene_percentile)

    def get_adj(self):
        """
        Obtain the adjacency matrix. The values in this adjacency matix has
        been scaled using regulatory norm. You may expect strong links to go
        beyond 5 or 10 in most cases.
        """
        return self.model.get_adj()


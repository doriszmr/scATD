import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
import torch.optim as optim
from optuna.pruners import MedianPruner
import logging
import torch.nn.init as init
from sklearn.model_selection import KFold
from torch.nn.utils import clip_grad_norm_
from sklearn.cluster import DBSCAN
from torch.nn.utils import weight_norm,spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')


parser = argparse.ArgumentParser(description="VAE Pretraining")


parser.add_argument('--open_path', type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in',
                    help='Path to the input data')
parser.add_argument('--save_path_outer', type=str,
                    default='/home/luozeyu/desktop/VAE_pretraining/output/Res_VAE_retraining_after_hyperparameter',
                    help='Outer path to save output results')

parser.add_argument('--file_prefix', type=str, default='scRNA-seq_panglao_0_1_Random_0_3',
                    help='File prefix for saving results')

parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--REC_beta', type=int, default=1000, help='Beta value for the REC loss')

parser.add_argument('--trial_num', type=int, default=100, help='')


parser.add_argument('--trail_study_new_name', type=str, help='')
parser.add_argument('--storge_trail_path', type=str, default="sqlite:////home/luozeyu/desktop/optuna_database/db1.sqlite3", help='creat new storge for optuna')
parser.add_argument('--device_choose', type=str, default='cuda:0', help='cpu choose')

parser.add_argument("--random_choose", type=strict_str2bool, default=True, help="")
parser.add_argument('--random_choose_num', type=float, default=0.3, help='')



args = parser.parse_args()



open_path = args.open_path
save_path_outer = args.save_path_outer
file_prefix = args.file_prefix

save_path = os.path.join(save_path_outer, file_prefix)
device_choose = args.device_choose



if not os.path.exists(save_path):
    os.makedirs(save_path)

random_choose_num = args.random_choose_num 


filename = os.listdir(open_path)[0]

scRNAseq_matrix_original = np.load(os.path.join(open_path, filename))


random_choose=args.random_choose 


print("DATA is containing NA?: ", np.isnan(scRNAseq_matrix_original).any())


scRNAseq_matrix = scRNAseq_matrix_original


if random_choose:
    num_samples = scRNAseq_matrix.shape[0]
    num_train_samples = int(num_samples * random_choose_num)  # 选择30%的数据作为训练集
    random_indices = np.random.choice(num_samples, size=num_train_samples, replace=False)

    # 使用随机索引选择训练集数据
    train_features = scRNAseq_matrix[random_indices,:]


# 定义DNN模型

# 启用异常检测
torch.autograd.set_detect_anomaly(True)


# 配置日志记录
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(save_path+r'\cross_validation_training_log.log'), logging.StreamHandler()])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(os.path.join(save_path, 'KF5_cross_validation_training_log.log')), logging.StreamHandler()])
logger = logging.getLogger()


# 训练函数
def train(model, device, train_loader, optimizer,REC_beta ):
    model.train()  # 设置模型为训练模式
    total_loss = 0  # 累计所有批次的损失

    for batch_idx, batch in enumerate(train_loader):
        # print(batch_idx)
        # print(batch)
        data = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar, beta=REC_beta)

        # 反向传播和优化
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=1.0)  # 将梯度的范数裁剪到1.0
        optimizer.step()

        # 累计损失
        total_loss += loss
    # 计算并返回平均损失
    avg_loss = total_loss / len(train_loader)

    return avg_loss


# 验证函数
def validate(model, device, val_loader,REC_beta):
    model.eval()  # 设置模型为评估模式
    total_loss = 0  # 累计所有批次的损失

    with torch.no_grad():  # 禁用梯度计算

        for batch_idx, batch in enumerate(val_loader):
            data = batch[0].to(device)
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, beta=REC_beta)

            # 累计损失
            total_loss += loss

    # 计算并返回平均损失
    avg_loss = total_loss / len(val_loader)

    return avg_loss



##vision3
import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))  # 可训练的β
        else:
            self.beta = initial_beta  # 固定的β

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# 可训练的β
# swish = Swish(trainable_beta=True)
# # 或固定β
# swish_fixed = Swish(trainable_beta=False, initial_beta=1.0)

class ContinuousResidualVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            # 先应用谱正则化
            self.fc = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
            # self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.swish = Swish(trainable_beta=True)  # 为每个 ResBlock 实例化独立的 Swish

            init.kaiming_normal_(self.fc.weight_orig, nonlinearity='leaky_relu')

            if in_dim != out_dim:

                self.downsample = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
                # self.downsample = nn.Linear(in_dim, out_dim)
                self.bn = nn.BatchNorm1d(out_dim)
                init.kaiming_normal_(self.downsample.weight_orig, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = self.swish(self.bn(self.fc(x)))

            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, input_dim,hidden_dim_layer0, Encoder_layer_dims,Decoder_layer_dims,hidden_dim_layer_out_Z, z_dim, loss_type='RMSE', reduction='sum'):
        super().__init__()
        # Encoder

        # Resblock
        self.Encoder_resblocks = nn.ModuleList()

        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))

        # # Latent space
       
        self.fc21 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=5)  # mu layer
        init.xavier_normal_(self.fc21.weight_orig)

        self.fc22 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim),
                                  n_power_iterations=5)  # logvariance layer
        init.xavier_normal_(self.fc22.weight_orig)


        # Decoder

        # Resblock
        self.Decoder_resblocks = nn.ModuleList()

        for i in range(len(Decoder_layer_dims) - 1):
            self.Decoder_resblocks.append(self.ResBlock(Decoder_layer_dims[i], Decoder_layer_dims[i + 1]))

        self.fc4 = spectral_norm(nn.Linear(hidden_dim_layer0, input_dim), n_power_iterations=5)
        init.xavier_normal_(self.fc4.weight_orig)

        # Add attributes for loss type and reduction type
        self.loss_type = loss_type
        self.reduction = reduction

        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)

    def encode(self, x):
        h = x
        for block in self.Encoder_resblocks:
            h = block(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h = F.leaky_relu(self.bn3(self.fc3(z)))
        h = z
        for block in self.Decoder_resblocks:
            h = block(h)
        return self.fc4(h)  # No sigmoid here

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):

        if self.loss_type == 'MSE':
            epsilon = 1e-8
            self.REC = F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction) + epsilon
        elif self.loss_type == 'RMSE':
            epsilon = 1e-8
            self.REC = torch.sqrt(F.mse_loss(recon_x, x.view(-1, x.shape[1]), reduction=self.reduction) + epsilon)
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

        if self.reduction == 'mean':
            self.KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            self.KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return beta * self.REC + self.KLD


    def get_model_inference_z(self, x):
        """
        This function takes input x and returns the corresponding latent vectors z.
        If a seed is provided, it is used to make the random number generator deterministic.
        """
        self.eval()  # switch to evaluation mode
        with torch.no_grad():  # disable gradient computation
            mu, logvar = self.encode(x.view(-1, x.shape[1]))
            z = self.reparameterize(mu, logvar)
        return z




# 定义分层交叉验证
kf =KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(X=train_features))

# # 提取标签用于分层采样
# labels = train_info_df['Label'].values


# 检查 CUDA 是否可用，并设置设备
device = torch.device(device_choose if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = args.batch_size
REC_beta =args.REC_beta

# hidden_dim_layer1 = 3072
# fold_min_loss =[]

def objective(trial):

    # 获取输入特征维度
    input_dim = train_features.shape[1]
    print(input_dim)
    #  # Define hyperparameters,超参数空间
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2  # , log=True
                                   )
    num_epochs = trial.suggest_int('num_epochs', 10, 60)
    z_dim = trial.suggest_int('z_dim', 150, 500)
    # hidden_dim = trial.suggest_int('hidden_dim', 500, 1000)
    hidden_dim_layer0 = trial.suggest_int('hidden_dim_layer0', 250, 2500)
    hidden_dim_layer_out_Z = trial.suggest_int('hidden_dim_layer_out_Z', 150, 500)

    num_blocks = trial.suggest_int('num_blocks', 1, 3)  # 动态选择残差块的数量

    Encoder_layer_dims =[input_dim]+ [hidden_dim_layer0] + [
        trial.suggest_int(f'layer_encoder_{i}_dim', 250, 1500, step=50) for i in range(1, num_blocks)
    ] + [hidden_dim_layer_out_Z]

    Decoder_layer_dims = [z_dim] + [
        trial.suggest_int(f'layer_decoder_{i}_dim', 250, 1500, step=50) for i in range(1, num_blocks)
    ] + [hidden_dim_layer0]

    # # 定义 Decoder 的层维度，与 Encoder 反向对称
    # Decoder_layer_dims = [hidden_dim_layer_out_Z] + Encoder_layer_dims[::-1][1:-2] + [input_dim]

    fold_min_loss = []
    #创建模型,
    for fold, (train_idx, val_idx) in enumerate(folds):

        # 对于 train_dataset
        train_features_sub = train_features[train_idx]

        train_dataset = TensorDataset(torch.Tensor(train_features_sub))

        # 对于 val_dataset
        val_features_sub = train_features[val_idx]
        val_dataset = TensorDataset(torch.Tensor(val_features_sub))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                drop_last=False)

        model = ContinuousResidualVAE(input_dim=input_dim, hidden_dim_layer0=hidden_dim_layer0,
                                      Encoder_layer_dims=Encoder_layer_dims, Decoder_layer_dims=Decoder_layer_dims,
                                      hidden_dim_layer_out_Z=hidden_dim_layer_out_Z, z_dim=z_dim, loss_type='MSE',
                                      reduction='mean').to(device)

        # 定义损失函数和优化器

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=int(num_epochs / 4), eta_min=learning_rate * (1e-2))
        min_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = train(model, device, train_loader, optimizer,REC_beta)
            val_loss = validate(model, device, val_loader,REC_beta)
            # 使用 logger 和 print 同时输出

            message = f"Trial {trial.number}, Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"

            logger.info(message)
            print(message)

            ## update learning rate
            scheduler.step()

            # Prune unpromising trials
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < min_loss and not torch.isinf(val_loss):
                min_loss = val_loss

            print(min_loss)
        fold_min_loss.append(min_loss)
        print(fold_min_loss)
        print(len(fold_min_loss))

    mean_min_loss = sum(fold_min_loss) / len(fold_min_loss)
    return mean_min_loss


# 创建和优化study
if __name__ == "__main__":

    study = optuna.create_study(direction='minimize', sampler=TPESampler(), pruner=MedianPruner(),study_name=args.trail_study_new_name,storage=args.storge_trail_path,load_if_exists=True)
    ##可以考虑使用多变量TPE估计，multivarate TPE
    # study = optuna.create_study(direction='minimize', sampler=TPESampler(multivariate=True), pruner=MedianPruner(),
    #                             study_name="scRNA-seq_panglao_ResVAE_pretraining_mission4",
    #                             storage="sqlite:///db.sqlite3", load_if_exists=True)

    #study = optuna.create_study(direction='minimize', sampler=TPESampler(), pruner=MedianPruner())

    study.optimize(objective, n_trials=args.trial_num)

    print('Best trial:')
    trial = study.best_trial
    print('Value:', trial.value)
    print('Params:')
    for key, value in trial.params.items():
        print(f'{key}: {value}')

    # 将最优超参数保存到 Excel 文件,改路径
    params_df = pd.DataFrame.from_dict(trial.params, orient='index', columns=['Value'])
    params_df.to_excel(os.path.join(save_path ,'best_hyperparameters.xlsx'))

    # 将试验数据转换为 DataFrame
    trials_df = pd.DataFrame([{
        'Trial Number': trial.number,
        'Value': trial.value,
        **trial.params  # 将字典的键值对展开为DataFrame的列
    } for trial in study.trials])

    # 导出
    trials_df.to_excel(os.path.join(save_path,'all_trial_hyperparameters.xlsx'), index=False)




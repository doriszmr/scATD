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
import matplotlib.pyplot as plt
import re
import argparse
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import CosineAnnealingLR

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
parser.add_argument('--open_path_conference_data', type=str,
                    default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data',
                    help='Path to the conference data')
parser.add_argument('--file_prefix', type=str, default='scRNA-seq_panglao_0_1_Random_0_3',
                    help='File prefix for saving results')
parser.add_argument("--best_parameter_name", type=str, default='best_hyperparameters.xlsx', help="File name for best hyperparameters.")
parser.add_argument('--epoch_start_for_loss_plot_only', type=int, default=1, help='Epoch start for loss plot only')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--REC_beta', type=int, default=1000, help='Beta value for the REC loss')
parser.add_argument('--device_choose', type=str, default='cuda:0', help='cpu choose')

parser.add_argument("--random_choose", type=strict_str2bool, default=True, help="")
parser.add_argument('--random_choose_num', type=float, default=0.3, help='')



args = parser.parse_args()

save_path = os.path.join(args.save_path_outer, args.file_prefix)


# Assign the arguments to variables
open_path = args.open_path
save_path_outer = args.save_path_outer
open_path_conference_data = args.open_path_conference_data
file_prefix = args.file_prefix
best_parameter_name = args.best_parameter_name
epoch_start_for_loss_plot_only = args.epoch_start_for_loss_plot_only
batch_size = args.batch_size
REC_beta = args.REC_beta
device_choose = args.device_choose
# Now you can use these variables in your script
print(f"Open Path: {open_path}")
print(f"Save Path Outer: {save_path_outer}")
print(f"Open Path Conference Data: {open_path_conference_data}")
print(f"File Prefix: {file_prefix}")
print(f"Best Parameter Name: {best_parameter_name}")
print(f"Save Path: {save_path}")
print(f"Epoch Start for Loss Plot Only: {epoch_start_for_loss_plot_only}")
print(f"Batch Size: {batch_size}")
print(f"REC Beta: {REC_beta}")



if not os.path.exists(save_path):
    os.makedirs(save_path)


random_choose_num = args.random_choose_num ##The proportion of data randomly selected for hyperparameter tuning.


filename = os.listdir(open_path)[0]

scRNAseq_matrix = np.load(os.path.join(open_path, filename))


random_choose=args.random_choose  


print("DATA is containing NA?: ", np.isnan(scRNAseq_matrix).any())



if random_choose:
    num_samples = scRNAseq_matrix.shape[0]
    num_train_samples = int(num_samples * random_choose_num)  
    random_indices = np.random.choice(num_samples, size=num_train_samples, replace=False)

    train_features = scRNAseq_matrix[random_indices,:]


torch.autograd.set_detect_anomaly(True)


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(save_path+r'\cross_validation_training_log.log'), logging.StreamHandler()])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(os.path.join(save_path, 'cross_validation_training_log.log')), logging.StreamHandler()])
logger = logging.getLogger()


def train(model, device, train_loader, optimizer,REC_beta):
    model.train()  
    total_loss = 0  

    for batch_idx, batch in enumerate(train_loader):
       
        data = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar, beta=REC_beta)

       
        loss.backward()
       
        optimizer.step()

        
        total_loss += loss

    avg_loss = total_loss / len(train_loader)

    return avg_loss



def validate(model, device, val_loader,REC_beta ):
    model.eval() 
    total_loss = 0  

    with torch.no_grad():  

        for batch_idx, batch in enumerate(val_loader):
            data = batch[0].to(device)
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, beta=REC_beta)

          
            total_loss += loss

  
    avg_loss = total_loss / len(val_loader)

    return avg_loss


import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta)) 
        else:
            self.beta = initial_beta  

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)



class ContinuousResidualVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
          
            self.fc = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
           
            self.bn = nn.BatchNorm1d(out_dim)
            self.swish = Swish(trainable_beta=True) 

            init.kaiming_normal_(self.fc.weight_orig, nonlinearity='leaky_relu')

            if in_dim != out_dim:

                self.downsample = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
                
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




kf =KFold(n_splits=10, shuffle=True, random_state=42)
folds = list(kf.split(X=train_features))


input_dim = train_features.shape[1]


best_params = pd.read_excel(os.path.join(open_path_conference_data, best_parameter_name ))
best_params = dict(zip(best_params.iloc[:,0], best_params.iloc[:,1]))

##capture the best hyperparamter
learning_rate = best_params['learning_rate']
weight_decay = best_params['weight_decay']
num_epochs = int(best_params['num_epochs'])
z_dim = int(best_params['z_dim'])
hidden_dim_layer0 = int(best_params['hidden_dim_layer0'])

hidden_dim_layer_out_Z = int(best_params['hidden_dim_layer_out_Z'])

num_blocks = best_params['num_blocks']


layer_encoder_dims = [int(best_params[f'layer_encoder_{i+1}_dim']) for i in range(int(num_blocks)-1)]

layer_decoder_dims = [int(best_params[f'layer_decoder_{i+1}_dim']) for i in range(int(num_blocks)-1)]


Encoder_layer_dims =[input_dim]+ [hidden_dim_layer0] + layer_encoder_dims + [hidden_dim_layer_out_Z]

Decoder_layer_dims = [z_dim] + layer_decoder_dims + [hidden_dim_layer0]



device = torch.device(device_choose if torch.cuda.is_available() else "cpu")
print("Using device:", device)



for fold, (train_idx, val_idx) in enumerate(folds):

   
    train_features_sub = train_features[train_idx]

    train_dataset = TensorDataset(torch.Tensor(train_features_sub))

 
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

 

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(num_epochs / 4), eta_min=learning_rate * (1e-2))

   

    for epoch in range(num_epochs):
        train_loss = train(model, device, train_loader, optimizer,REC_beta)
        val_loss = validate(model, device, val_loader,REC_beta)
       

        message = f" Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"

        logger.info(message)
        print(message)

        ## update learning rate
        scheduler.step()


        if (epoch + 1) % 3 == 0:
            checkpoint_path = os.path.join(save_path,f'checkpoint_fold{fold + 1}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)




# Function to extract log data from a logger file
def extract_log_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.readlines()

    data = []
    pattern = re.compile(r"Fold (\d+), Epoch (\d+)/(\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)")

    for line in log_data:
        match = pattern.search(line)
        if match:
            fold = int(match.group(1))
            epoch = int(match.group(2))
            train_loss = float(match.group(4))
            val_loss = float(match.group(5))
            data.append([fold, epoch, train_loss, val_loss])

    return pd.DataFrame(data, columns=['Fold', 'Epoch', 'Train Loss', 'Val Loss'])


# Path to the logger file
file_path = (os.path.join(save_path, 'cross_validation_training_log.log') ) # Replace with the actual path to your logger file

# Extract log data
df = extract_log_data(file_path)

# Remove the first epoch from each fold
df = df[df['Epoch'] > epoch_start_for_loss_plot_only]
print('epoch_start_for_loss_plot_only:', epoch_start_for_loss_plot_only)
# Calculate mean and standard deviation for each epoch across folds
train_loss_mean = df.groupby('Epoch')['Train Loss'].mean()
train_loss_std = df.groupby('Epoch')['Train Loss'].std()
val_loss_mean = df.groupby('Epoch')['Val Loss'].mean()
val_loss_std = df.groupby('Epoch')['Val Loss'].std()

# Plotting the train and validation loss curves
plt.figure(figsize=(12, 6))

# Plot Train Loss
plt.subplot(1, 2, 1)
plt.plot(train_loss_mean.index, train_loss_mean, label='Train Loss')
plt.fill_between(train_loss_mean.index, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(val_loss_mean.index, val_loss_mean, label='Validation Loss', color='orange')
plt.fill_between(val_loss_mean.index, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2,
                 color='orange')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plot as a PDF
plt.savefig(os.path.join(save_path,f"{file_prefix}model_Loss_plot.pdf"),dpi = 1000, format='pdf')


# Plotting the combined train and validation loss curves
plt.figure(figsize=(8, 6))

plt.plot(train_loss_mean.index, train_loss_mean, label='Train Loss')
plt.fill_between(train_loss_mean.index, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
plt.plot(val_loss_mean.index, val_loss_mean, label='Validation Loss', color='orange')
plt.fill_between(val_loss_mean.index, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color='orange')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

# Save the plot as a PDF
plt.savefig(os.path.join(save_path,f"{file_prefix}combined_model_Loss_plot.pdf"),dpi = 1000, format='pdf')



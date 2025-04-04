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


open_path = '/home/luozeyu/desktop/VAE_pretraining/data/in'
save_path_outer = '/home/luozeyu/desktop/VAE_pretraining/output/'

file_prefix = 'scRNA-seq_panglao_0_1_Random_0_3'

save_path = os.path.join(save_path_outer, file_prefix)


if not os.path.exists(save_path):
    os.makedirs(save_path)

random_choose_num = 0.3


filename = os.listdir(open_path)[0]

scRNAseq_matrix_original = np.load(os.path.join(open_path, filename))


random_choose=True  #or False if in less data



print("DATA is containing NA?: ", np.isnan(scRNAseq_matrix_original).any())


scRNAseq_matrix = scRNAseq_matrix_original



if random_choose:
    num_samples = scRNAseq_matrix.shape[0]
    num_train_samples = int(num_samples * random_choose_num)  
    random_indices = np.random.choice(num_samples, size=num_train_samples, replace=False)

    train_features = scRNAseq_matrix[random_indices,:]


torch.autograd.set_detect_anomaly(True)


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(save_path+r'\cross_validation_training_log.log'), logging.StreamHandler()])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(os.path.join(save_path, 'KF5_cross_validation_training_log.log')), logging.StreamHandler()])
logger = logging.getLogger()



def train(model, device, train_loader, optimizer,REC_beta ):
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



def validate(model, device, val_loader,REC_beta):
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

class ContinuousResidualVAE(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)
            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
            self.bn = nn.BatchNorm1d(out_dim)

            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = F.leaky_relu(self.bn(self.fc(x)))

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

        # Latent space
        self.fc21 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # mu layer
        init.xavier_normal_(self.fc21.weight)  # Xavier Initialization for mu layer
        self.fc22 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # logvariance layer
        init.xavier_normal_(self.fc22.weight)  # Xavier Initialization for logvar layer

        # Decoder

       

        # Resblock
        self.Decoder_resblocks = nn.ModuleList()

        for i in range(len(Decoder_layer_dims) - 1):
            self.Decoder_resblocks.append(self.ResBlock(Decoder_layer_dims[i], Decoder_layer_dims[i + 1]))

        self.fc4 = nn.Linear(hidden_dim_layer0, input_dim)
        init.xavier_normal_(self.fc4.weight)
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


    def get_model_inference_z(self, x, seed=None):
        """
        This function takes input x and returns the corresponding latent vectors z.
        If a seed is provided, it is used to make the random number generator deterministic.
        """
        self.eval()  # switch to evaluation mode
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():  # disable gradient computation
            mu, logvar = self.encode(x.view(-1, x.shape[1]))
            z = self.reparameterize(mu, logvar)
        return z


kf =KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(X=train_features))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 128
REC_beta =1000
# hidden_dim_layer1 = 3072
fold_min_loss =[]

def objective(trial):

  
    input_dim = train_features.shape[1]

   
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2  # , log=True
                                   )
    num_epochs = trial.suggest_int('num_epochs', 10, 60)
    z_dim = trial.suggest_int('z_dim', 200, 500)
    # hidden_dim = trial.suggest_int('hidden_dim', 500, 1000)
    hidden_dim_layer0 = trial.suggest_int('hidden_dim_layer0', 1500, 2500)
    hidden_dim_layer_out_Z = trial.suggest_int('hidden_dim_layer_out_Z', 250, 500)

    num_blocks = trial.suggest_int('num_blocks', 1, 3) 

    Encoder_layer_dims =[input_dim]+ [hidden_dim_layer0] + [
        trial.suggest_int(f'layer_encoder_{i}_dim', 750, 1500, step=50) for i in range(1, num_blocks)
    ] + [hidden_dim_layer_out_Z]

    Decoder_layer_dims = [z_dim] + [
        trial.suggest_int(f'layer_decoder_{i}_dim', 750, 1500, step=50) for i in range(1, num_blocks)
    ] + [hidden_dim_layer0]


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

        min_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = train(model, device, train_loader, optimizer,REC_beta)
            val_loss = validate(model, device, val_loader,REC_beta)
          

            message = f"Trial {trial.number}, Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"

            logger.info(message)
            print(message)

            # Prune unpromising trials
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < min_loss and not torch.isinf(val_loss):
                min_loss = val_loss

        fold_min_loss.append(min_loss)

    mean_min_loss = sum(fold_min_loss) / len(fold_min_loss)
    return mean_min_loss



if __name__ == "__main__":

    study = optuna.create_study(direction='minimize', sampler=TPESampler(), pruner=MedianPruner(),study_name="scRNA-seq_panglao_ResVAE_pretraining_mission4",storage="sqlite:///db.sqlite3",load_if_exists=True)
   
    study.optimize(objective, n_trials=100)

    print('Best trial:')
    trial = study.best_trial
    print('Value:', trial.value)
    print('Params:')
    for key, value in trial.params.items():
        print(f'{key}: {value}')

 
    params_df = pd.DataFrame.from_dict(trial.params, orient='index', columns=['Value'])
    params_df.to_excel(os.path.join(save_path ,'best_hyperparameters.xlsx'))

    
    trials_df = pd.DataFrame([{
        'Trial Number': trial.number,
        'Value': trial.value,
        **trial.params 
    } for trial in study.trials])


    trials_df.to_excel(os.path.join(save_path,'all_trial_hyperparameters.xlsx'), index=False)




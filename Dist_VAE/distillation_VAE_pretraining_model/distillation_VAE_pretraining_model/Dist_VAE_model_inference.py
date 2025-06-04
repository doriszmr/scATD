

import torch
from torch import nn
from torch.nn.utils import spectral_norm
import torch.nn.init as init
import numpy as np

## Swish function definition
class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))
        else:
            self.beta = initial_beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

swish = Swish(trainable_beta=True)

## Dist VAE model definition

class VAEclassification(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            #self.fc = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
            self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.swish = swish

            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')

            if in_dim != out_dim:

                self.downsample = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
                self.bn = nn.BatchNorm1d(out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = self.swish(self.bn(self.fc(x)))

            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, Encoder_layer_dims, hidden_dim_layer_out_Z, z_dim, class_num):
        super(VAEclassification, self).__init__()
        # Encoder

        # Resblock
        self.Encoder_resblocks = nn.ModuleList()

        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))


        self.fc21 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=5)  # mu layer
        init.xavier_normal_(self.fc21.weight)

        self.fc22 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim),
                                      n_power_iterations=5)  # logvariance layer
        init.xavier_normal_(self.fc22.weight)

        self.fc3 = spectral_norm(nn.Linear(z_dim, z_dim // 2), n_power_iterations=5)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        self.bn3 = nn.BatchNorm1d(z_dim // 2)
        self.swish = swish
        self.fc4 = spectral_norm(nn.Linear(z_dim // 2, class_num), n_power_iterations=5)
        init.xavier_normal_(self.fc4.weight)


    def encode(self, x):
        h = x
        for block in self.Encoder_resblocks:
            h = block(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classifier(self, x):
        h = self.fc3(x)
        h = self.bn3(h)
        h = self.swish(h)
        h = self.fc4(h)
        return h


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        class_logits = self.classifier(z)
        return class_logits,z




def modelevalution(model, loader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            output,_ = model(data)
            all_predictions.extend(output.softmax(dim=1)[:, 1].tolist())  # 针对二分类任务的第二类的概率
            all_targets.extend(target.tolist())
    return all_predictions, all_targets


def modelinference(model, loader):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for data in loader:
            output,_ = model(data[0])
            all_predictions.extend(output.softmax(dim=1)[:, 1].tolist())  # 针对二分类任务的第二类的概率
    return all_predictions



def z_embedding(model, loader):
    model.eval()
    z_list = []
    with torch.no_grad():
        for data, _ in loader:
            _, z = model(data)
            z_list.append(z.cpu().numpy())
    z_numpy = np.concatenate(z_list, axis=0)
    return z_numpy


##vision3
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm


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



class ContinuousResidualVAE(nn.Module):
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
                # self.downsample = nn.Linear(in_dim, out_dim)
                self.bn = nn.BatchNorm1d(out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
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

        self.fc21 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=5)  # mu layer
        init.xavier_normal_(self.fc21.weight)

        self.fc22 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim),
                                  n_power_iterations=5)  # logvariance layer
        init.xavier_normal_(self.fc22.weight)


        # Decoder
        # Resblock
        self.Decoder_resblocks = nn.ModuleList()

        for i in range(len(Decoder_layer_dims) - 1):
            self.Decoder_resblocks.append(self.ResBlock(Decoder_layer_dims[i], Decoder_layer_dims[i + 1]))

        self.fc4 = spectral_norm(nn.Linear(hidden_dim_layer0, input_dim), n_power_iterations=5)

        init.xavier_normal_(self.fc4.weight)
        # Add attributes for loss type and reduction type

        self.loss_type = loss_type
        self.reduction = reduction

        if reduction not in ['mean', 'sum']:
            raise ValueError("Invalid reduction type. Expected 'mean' or 'sum', but got %s" % reduction)

    def encode(self, x):
        h = x
        for i, block in enumerate(self.Encoder_resblocks):
            h = block(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = z
        for block in self.Decoder_resblocks:
            h = block(h)
        return self.fc4(h)  # No sigmoid here

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar,z

    def cos_loss(self,x, y, reduction='mean'):

        similarity = F.cosine_similarity(x, y, dim=1)

        loss = 1 - similarity

        if reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("Invalid value for reduction: {}".format(reduction))

    def loss_function(self, recon_x, x, recon_scVAE, x_scVAE, mu, logvar, beta=1.0,cosine_beta=1.0):

        self.REC_scVAE = self.cos_loss(recon_scVAE, x_scVAE.view(-1, x_scVAE.shape[1]), reduction=self.reduction)


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

        return beta*self.REC + cosine_beta*self.REC_scVAE + self.KLD


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




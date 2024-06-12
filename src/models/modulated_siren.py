import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import wandb

# Modulation size is 256

class ModulatedSineLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, modulation_size=256):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.modulation = nn.Linear(modulation_size, out_features, bias=True)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input, phi):
        return torch.sin(self.omega_0 * (self.linear(input) + self.modulation(phi)))

class ModulatedSirenModel(pl.LightningModule):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.phi = nn.Parameter(torch.zeros(256))
        self.last_used_optimizer_index = 0
        self.net = []
        self.net.append(ModulatedSineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(ModulatedSineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(ModulatedSineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        for layer in self.net[:-1]:
            x = layer(x, self.phi)
        x = self.net[-1](x)
        return x

    def on_train_epoch_start(self):
        if self.current_epoch % 4 == 0:
            self.last_used_optimizer_index = 0
            self.phi.data = torch.zeros(256).cuda()
            self.phi.requires_grad = False
            for layer in self.net[:-1]:
                layer.linear.weight.requires_grad = True
                layer.linear.bias.requires_grad = True
                layer.modulation.weight.requires_grad = True
                layer.modulation.bias.requires_grad = True
            self.net[-1].weight.requires_grad = True
            self.net[-1].bias.requires_grad = True
        else:
            self.last_used_optimizer_index = 1
            self.phi.requires_grad = True
            for layer in self.net[:-1]:
                layer.linear.weight.requires_grad = False
                layer.linear.bias.requires_grad = False
                layer.modulation.weight.requires_grad = False
                layer.modulation.bias.requires_grad = False
            self.net[-1].weight.requires_grad = False
            self.net[-1].bias.requires_grad = False
        
        print(f"START EPOCH WEIGHT: {self.net[-2].linear.weight}")
        print(f"START EPOCH BIAS: {self.net[-2].linear.bias}")
        print(f"START EPOCH phi: {self.phi}")

    def training_step(self, batch, batch_idx, optimizer_idx):
        coords, pixels = batch
        pixels_hat = self(coords)
        loss = torch.nn.functional.mse_loss(pixels_hat, pixels)
        self.log('train_loss', loss)
        return loss
    
    def on_train_epoch_end(self):
        


        optimizer = self.trainer.optimizers[self.last_used_optimizer_index]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr)
        print(f"END EPOCH WEIGHT: {self.net[-2].linear.weight}")
        print(f"END EPOCH BIAS: {self.net[-2].linear.bias}")
        print(f"END EPOCH phi: {self.phi}")
        
        # self.log('phi', self.phi)
        self.log('global_gradient_step', self.global_step)
    
    def on_train_end(self):
        print('Training finished!')

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.parameters(), lr=3e-6)
        optimizer2 = torch.optim.SGD(self.parameters(), lr=1e-2)
        return [optimizer1, optimizer2]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if epoch % 4 == 0:
            self.last_used_optimizer_index = 0
            optimizer = self.optimizers()[0]  # Use optimizer1
        else:
            self.last_used_optimizer_index = 1
            optimizer = self.optimizers()[1]  # Use optimizer2         


        # What the fuck is optimizer_closure???
        optimizer.step(closure=optimizer_closure)


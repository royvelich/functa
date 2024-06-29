import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from tqdm import tqdm
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
    def __init__(self, in_features, hidden_features, hidden_layers, modulation_size, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., lr=3e-6, epochs=2000):
        super().__init__()

        self.lr = lr
        self.epochs = epochs
        self.modulation_size = modulation_size
        self.phi = nn.Parameter(torch.zeros(modulation_size))
        
        self.net = []
        self.net.append(ModulatedSineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0, modulation_size=self.modulation_size))

        for i in range(hidden_layers):
            self.net.append(ModulatedSineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0, modulation_size=modulation_size))

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

    def freeze_phi(self):
        self.phi.requires_grad = False
        for layer in self.net[:-1]:
            layer.linear.weight.requires_grad = True
            layer.linear.bias.requires_grad = True
            layer.modulation.weight.requires_grad = True
            layer.modulation.bias.requires_grad = True
        self.net[-1].weight.requires_grad = True
        self.net[-1].bias.requires_grad = True
    
    def freeze_base(self):
        self.phi.data = torch.zeros(self.modulation_size).to(self.device)
        self.phi.requires_grad = True
        for layer in self.net[:-1]:
            layer.linear.weight.requires_grad = False
            layer.linear.bias.requires_grad = False
            layer.modulation.weight.requires_grad = False
            layer.modulation.bias.requires_grad = False
        self.net[-1].weight.requires_grad = False
        self.net[-1].bias.requires_grad = False
    
    def on_train_batch_start(self, batch, batch_idx):
        self.freeze_base()
        inner_loop_optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        for _ in range(3):
            inner_loop_optimizer.zero_grad()
            coords, pixels = batch
            pixels_hat = self(coords)
            loss = torch.nn.functional.mse_loss(pixels_hat, pixels)
            loss.backward()
            inner_loop_optimizer.step()
        self.log('inner_loop_loss', loss)
        self.freeze_phi()

    def training_step(self, batch, batch_idx):
        coords, pixels = batch
        pixels_hat = self(coords)
        loss = torch.nn.functional.mse_loss(pixels_hat, pixels)
        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr)
        self.log('global_gradient_step', self.global_step)
    
    def on_train_end(self):
        print('Training finished!')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # milestones = [int(self.epochs * 0.3), int(self.epochs * 0.7)]
        # scheduler = MultiStepLR(optimizer1, milestones=milestones, gamma=0.1)
        
        # optimizer2 = torch.optim.SGD(self.parameters(), lr=1e-2)
        # return {
        #     'optimizer': optimizer1,
        #     'lr_scheduler': scheduler,
        # }
        return [optimizer]

    def train_latent(self, batch):
        self.phi.data = torch.zeros(self.modulation_size)
        self.phi.requires_grad = True
        for layer in self.net[:-1]:
            layer.linear.weight.requires_grad = False
            layer.linear.bias.requires_grad = False
            layer.modulation.weight.requires_grad = False
            layer.modulation.bias.requires_grad = False
        self.net[-1].weight.requires_grad = False
        self.net[-1].bias.requires_grad = False



        inner_loop_optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        for _ in tqdm(range(3)):
            print(f"START PHI: {self.phi}")
            inner_loop_optimizer.zero_grad()
            coords, pixels = batch
            pixels_hat = self(coords)
            loss = torch.nn.functional.mse_loss(pixels_hat, pixels)
            loss.backward()
            inner_loop_optimizer.step()
            print(f"END PHI: {self.phi}")


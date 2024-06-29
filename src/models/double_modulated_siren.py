import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from tqdm import tqdm
from src.models.modulated_siren import ModulatedSineLayer
import wandb

class DoubleModulatedSirenModel(pl.LightningModule):
    def __init__(self, in_features, hidden_features, hidden_layers, modulation_size, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., lr=3e-6, epochs=2000):
        super().__init__()

        self.lr = lr
        self.epochs = epochs
        self.modulation_size = modulation_size
        self.phi = nn.Parameter(torch.zeros(modulation_size))

        self.net_1 = self.build_net(in_features, hidden_features, hidden_layers, out_features, outermost_linear, first_omega_0, hidden_omega_0)
        self.net_2 = self.build_net(in_features, hidden_features, hidden_layers, out_features, outermost_linear, first_omega_0, hidden_omega_0)
        
    
    def build_net(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear, first_omega_0, hidden_omega_0):
        net = []
        net.append(ModulatedSineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, modulation_size=self.modulation_size))

        for i in range(hidden_layers):
            net.append(ModulatedSineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, modulation_size=self.modulation_size))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            net.append(final_linear)
        else:
            net.append(ModulatedSineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        return nn.Sequential(*net)

    def forward_net(self, net, x):
        # print(f"FORWARD NET DEVICES {x.device}")
        # print(f"PHI DEVICE {self.phi.device}")
        for i, layer in enumerate(net[:-1]):
            # print(f"FORWARD LAYER {i} DEVICES {x.device}")

            x = layer(x, self.phi)
        x = net[-1](x)
        return x

    def forward(self, x):
        x_1, x_2 = x
        # print(f"FORWARD DEVICES {x_1.device, x_2.device}")
        return self.forward_net(self.net_1, x_1), self.forward_net(self.net_2, x_2)

    def freeze_phi(self):
        self.phi.requires_grad = False
        for net in [self.net_1, self.net_2]:
            for layer in net[:-1]:
                layer.linear.weight.requires_grad = True
                layer.linear.bias.requires_grad = True
                layer.modulation.weight.requires_grad = True
                layer.modulation.bias.requires_grad = True
            net[-1].weight.requires_grad = True
            net[-1].bias.requires_grad = True
    
    def freeze_base(self):
        self.phi.data = torch.zeros(self.modulation_size).to(self.device)
        self.phi.requires_grad = True
        for net in [self.net_1, self.net_2]:
            for layer in net[:-1]:
                layer.linear.weight.requires_grad = False
                layer.linear.bias.requires_grad = False
                layer.modulation.weight.requires_grad = False
                layer.modulation.bias.requires_grad = False
            net[-1].weight.requires_grad = False
            net[-1].bias.requires_grad = False

    def on_train_batch_start(self, batch, batch_idx):
        self.freeze_base()
        # print("ON BATCH STAAAAAART")
        inner_loop_optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        for _ in range(3):
            inner_loop_optimizer.zero_grad()
            coords_1, pixels_1, coords_2, pixels_2 = batch
            pixels_1_hat, pixels_2_hat = self.forward((coords_1, coords_2))
            loss_1 = torch.nn.functional.mse_loss(pixels_1_hat, pixels_1)
            loss_2 = torch.nn.functional.mse_loss(pixels_2_hat, pixels_2)
            loss = (loss_1 + loss_2) / 2
            loss.backward()
            inner_loop_optimizer.step()

            # Need to take mean of phi here.

        self.log('inner_loop_loss', loss)
        self.freeze_phi()


    def training_step(self, batch, batch_idx):
        coords_1, pixels_1, coords_2, pixels_2 = batch
        pixels_1_hat, pixels_2_hat = self.forward((coords_1, coords_2))

        loss_1 = torch.nn.functional.mse_loss(pixels_1_hat, pixels_1)
        loss_2 = torch.nn.functional.mse_loss(pixels_2_hat, pixels_2)

        loss = (loss_1 + loss_2) / 2
        self.log('main loop loss', loss)

        return loss

    def train_latent(self, batch):
        self.phi.data = torch.zeros(self.modulation_size)
        self.phi.requires_grad = True

        for net in [self.net_1, self.net_2]:
            for layer in net[:-1]:
                layer.linear.weight.requires_grad = False
                layer.linear.bias.requires_grad = False
                layer.modulation.weight.requires_grad = False
                layer.modulation.bias.requires_grad = False
            net[-1].weight.requires_grad = False
            net[-1].bias.requires_grad = False
        
        inner_loop_optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        for _ in range(3):
            inner_loop_optimizer.zero_grad()
            coords_1, pixels_1, coords_2, pixels_2 = batch
            pixels_1_hat, pixels_2_hat = self.forward((coords_1, coords_2))
            loss_1 = torch.nn.functional.mse_loss(pixels_1_hat, pixels_1)
            loss_2 = torch.nn.functional.mse_loss(pixels_2_hat, pixels_2)
            loss = (loss_1 + loss_2) / 2
            loss.backward()
            inner_loop_optimizer.step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return [optimizer]
        
import torch
import pytorch_lightning as pl
import numpy as np

class SirenLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, w0=30, first_layer=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = torch.nn.Linear(in_features, out_features)
        self.w0 = w0

        if first_layer:
            self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
        else:
            self.linear.weight.uniform_(-np.sqrt(6 / in_features) / w0, np.sqrt(6 / in_features) / w0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class SirenModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = SirenLayer(1, 50, first_layer=True)
        self.layer2 = SirenLayer(50, 50)
        self.layer3 = SirenLayer(50, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

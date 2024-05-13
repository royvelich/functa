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

        with torch.no_grad():
            if first_layer:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / w0, np.sqrt(6 / in_features) / w0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class SirenModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        layer_sizes = [2] + [256] * 9 + [3]
        self.layers = torch.nn.ModuleList(
            [SirenLayer(layer_sizes[i], layer_sizes[i + 1], first_layer=(i == 0)) for i in range(len(layer_sizes) - 1)]
        )

        print("Model initialized")
        print(f"Layer 1: {self.layers[0]}")
        print(f"Bias: {self.layers[0].linear.bias}")
        print(f"Bias_len: {len(self.layers[0].linear.bias)}")
        print(f"Bias shape: {self.layers[0].linear.bias.shape}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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

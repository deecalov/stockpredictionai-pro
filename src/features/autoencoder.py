import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple

class GELU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)

class StackedAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, bottleneck: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            GELU(),
            nn.Linear(hidden, bottleneck),
            GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            GELU(),
            nn.Linear(hidden, in_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class VariationalAutoencoder(nn.Module):
    """VAE variant matching the source notebook (encoder -> mu/logvar + KL term).

    Inputs are standardized internally (mean/std buffers fitted on train data)
    so the KL term stays numerically stable on raw, unscaled features.
    Latent features extracted at inference are the deterministic mu vector;
    the reparameterization trick (mu + eps * sigma) is used only in training.
    """

    def __init__(self, in_dim: int, hidden: int = 64, bottleneck: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            GELU(),
        )
        self.enc_out = nn.Linear(hidden, bottleneck * 2)  # -> (mu, logvar)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            GELU(),
            nn.Linear(hidden, in_dim),
        )
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.x_mean.copy_(mean)
        self.x_std.copy_(std.clamp_min(1e-8))

    def normalize(self, x):
        return (x - self.x_mean) / self.x_std

    def encode(self, x):
        h = self.enc_out(self.encoder(self.normalize(x)))
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, x):
        """Returns (recon of the *normalized* input, mu, logvar)."""
        mu, logvar = self.encode(x)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        recon = self.decoder(z)
        return recon, mu, logvar


def _build_ae(variant: str, in_dim: int, hidden: int, bottleneck: int) -> nn.Module:
    if variant == "stacked":
        return StackedAutoencoder(in_dim, hidden, bottleneck)
    if variant == "vae":
        return VariationalAutoencoder(in_dim, hidden, bottleneck)
    raise ValueError(f"Unknown autoencoder variant: {variant!r} (use 'stacked' or 'vae')")


def _extract_latent(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if isinstance(model, VariationalAutoencoder):
        mu, _ = model.encode(xb)
        return mu
    _, z = model(xb)
    return z


def fit_autoencoder(df: pd.DataFrame, hidden=64, bottleneck=32, epochs=10,
                    batch_size=128, lr=1e-3, device=None,
                    variant="stacked", vae_beta=1e-3,
                    xavier_init=True) -> Tuple[pd.DataFrame, nn.Module]:
    """Train an autoencoder on df and return (latent features, model).

    Args:
        variant: "stacked" (deterministic AE) or "vae" (variational, KL term).
        vae_beta: weight of the KL term in the VAE loss.
        xavier_init: apply Xavier initialization (as in the source notebook).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(df.values, dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = _build_ae(variant, df.shape[1], hidden, bottleneck).to(device)
    if xavier_init:
        from ..models.base import init_weights_xavier
        init_weights_xavier(model)
    if isinstance(model, VariationalAutoencoder):
        model.set_normalization(X.mean(dim=0).to(device),
                                X.std(dim=0).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            if isinstance(model, VariationalAutoencoder):
                recon, mu, logvar = model(xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss_fn(recon, model.normalize(xb)) + vae_beta * kl
            else:
                recon, _ = model(xb)
                loss = loss_fn(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
    # extract features
    model.eval()
    with torch.no_grad():
        z = []
        for (xb,) in DataLoader(ds, batch_size=batch_size):
            zi = _extract_latent(model, xb.to(device))
            z.append(zi.cpu().numpy())
    Z = np.vstack(z)
    cols = [f"ae_{i:02d}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, index=df.index, columns=cols), model


def transform_autoencoder(df: pd.DataFrame, model: nn.Module, batch_size=128, device=None) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(df.values, dtype=torch.float32)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    Zs = []
    with torch.no_grad():
        for (xb,) in dl:
            zi = _extract_latent(model, xb.to(device))
            Zs.append(zi.cpu().numpy())
    Z = np.vstack(Zs)
    cols = [f"ae_{i:02d}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, index=df.index, columns=cols)

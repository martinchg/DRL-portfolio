# diffusion/model.py
"""
Réseau de débruitage : U-Net 1D convolutif.

Entrée (B, 1, L) = fenêtre bruitée x_t + timestep de diffusion t ;
sortie (B, 1, L) = bruit ε prédit. L'architecture voit la fenêtre à trois
résolutions (L, L/2, L/4) : les échelles grossières portent la structure
lente (régimes de volatilité), les fines le grain journalier.
"""
import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encodage sinusoïdal du pas de diffusion t (comme les Transformers)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock1D(nn.Module):
    """Bloc résiduel Conv1d ; le timestep est injecté entre les deux convs."""

    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, c_in), c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.temb = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.skip = (nn.Conv1d(c_in, c_out, 1)
                     if c_in != c_out else nn.Identity())
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb(self.act(temb))[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """
    3 niveaux (canaux ex. 32→64→128), 2 downsamplings ×2, skips symétriques.
    Conv finale initialisée à zéro : le réseau démarre en prédisant ε ≈ 0,
    ce qui stabilise les premiers pas d'entraînement.
    """

    def __init__(self, channels=(32, 64, 128), t_dim: int = 128):
        super().__init__()
        c0, c1, c2 = channels
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        self.in_conv = nn.Conv1d(1, c0, 3, padding=1)

        # Descente
        self.d0a, self.d0b = ResBlock1D(c0, c0, t_dim), ResBlock1D(c0, c0, t_dim)
        self.down0 = nn.Conv1d(c0, c1, 3, stride=2, padding=1)
        self.d1a, self.d1b = ResBlock1D(c1, c1, t_dim), ResBlock1D(c1, c1, t_dim)
        self.down1 = nn.Conv1d(c1, c2, 3, stride=2, padding=1)

        # Fond
        self.mid_a, self.mid_b = ResBlock1D(c2, c2, t_dim), ResBlock1D(c2, c2, t_dim)

        # Remontée (upsample + conv, concat du skip, résblocs)
        self.up1 = nn.Conv1d(c2, c1, 3, padding=1)
        self.u1a, self.u1b = ResBlock1D(c1 * 2, c1, t_dim), ResBlock1D(c1, c1, t_dim)
        self.up0 = nn.Conv1d(c1, c0, 3, padding=1)
        self.u0a, self.u0b = ResBlock1D(c0 * 2, c0, t_dim), ResBlock1D(c0, c0, t_dim)

        self.out_norm = nn.GroupNorm(min(8, c0), c0)
        self.out_conv = nn.Conv1d(c0, 1, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.act = nn.SiLU()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t)

        h = self.in_conv(x)
        h = self.d0a(h, temb)
        s0 = self.d0b(h, temb)              # skip pleine résolution
        h = self.down0(s0)
        h = self.d1a(h, temb)
        s1 = self.d1b(h, temb)              # skip résolution L/2
        h = self.down1(s1)

        h = self.mid_a(h, temb)
        h = self.mid_b(h, temb)

        h = self.up1(self.upsample(h))
        h = self.u1a(torch.cat([h, s1], dim=1), temb)
        h = self.u1b(h, temb)
        h = self.up0(self.upsample(h))
        h = self.u0a(torch.cat([h, s0], dim=1), temb)
        h = self.u0b(h, temb)

        return self.out_conv(self.act(self.out_norm(h)))

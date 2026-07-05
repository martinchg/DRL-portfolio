# diffusion/schedule.py
"""
Processus de diffusion avant (forward) : bruitage progressif en T pas.

Tout tient dans les ᾱ_t (produits cumulés des 1-β_t) :
    q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t) · I)
d'où le tirage en forme fermée q_sample — pas besoin d'itérer les T pas
pendant l'entraînement, on tire t au hasard et on bruite d'un coup.

Schedule cosine (Nichol & Dhariwal 2021) par défaut : détruit l'information
plus progressivement qu'un schedule linéaire aux premiers pas — mieux adapté
aux signaux de faible dimension comme nos fenêtres 1D.
"""
import math
from typing import Optional

import torch


def make_betas(T: int, kind: str = "cosine", s: float = 0.008) -> torch.Tensor:
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, T)
    if kind == "cosine":
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(max=0.999).float()
    raise ValueError(f"Schedule inconnu : {kind}")


class DiffusionSchedule:
    """Pré-calcule tous les coefficients du forward ET du reverse process."""

    def __init__(self, T: int = 1000, kind: str = "cosine",
                 device: str = "cpu"):
        self.T = T
        self.kind = kind
        betas = make_betas(T, kind).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=device), alpha_bars[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        # Variance de la posterior q(x_{t-1} | x_t, x_0) — utilisée au sampling
        self.posterior_var = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

    def q_sample(
        self,
        x0    : torch.Tensor,          # (B, 1, L)
        t     : torch.Tensor,          # (B,) long
        noise : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε — bruitage direct au pas t."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

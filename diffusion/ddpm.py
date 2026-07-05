# diffusion/ddpm.py
"""
DDPM : loss d'entraînement, sampling ancestral, EMA des poids.

Entraînement (Ho et al. 2020, objectif simplifié) :
    t ~ U[0, T) ; ε ~ N(0, I) ; x_t = q_sample(x_0, t, ε)
    loss = ‖ε − ε_θ(x_t, t)‖²

Sampling ancestral (reverse process) :
    x_{t-1} = 1/√α_t · (x_t − β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + √σ²_t · z

L'EMA (moyenne mobile exponentielle des poids) est le modèle QU'ON ÉVALUE :
les poids instantanés oscillent avec les derniers minibatches, la moyenne
lissée échantillonne nettement mieux — standard dans toute la littérature.
"""
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from diffusion.model import UNet1D
from diffusion.schedule import DiffusionSchedule


class EMA:
    """Moyenne mobile exponentielle des paramètres du modèle."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: p.detach().clone()
            for k, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[k].mul_(self.decay).add_(p, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for k, p in model.named_parameters():
            if k in self.shadow:
                p.copy_(self.shadow[k])

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, device: str = "cpu") -> None:
        self.shadow = {k: v.to(device) for k, v in sd.items()}


class DDPM:
    """
    pred_type :
    - 'eps' : le réseau prédit le bruit ε (Ho et al. 2020).
    - 'v'   : le réseau prédit v = √ᾱ·ε − √(1-ᾱ)·x₀ (Salimans & Ho 2022).
      Mieux conditionné à tous les niveaux de bruit. Important ici : les
      rendements sont quasi BLANCS (signal et bruit ont la même texture), un
      ε̂ à peine biaisé sur-attribue du signal au bruit et le biais se COMPOSE
      sur les T pas du reverse process → variance échantillonnée rétrécie
      (mesuré en v1 : z-std 0.52 au lieu de 1).
    """

    def __init__(self, model: UNet1D, schedule: DiffusionSchedule,
                 device: str = "cpu", pred_type: str = "eps"):
        if pred_type not in ("eps", "v"):
            raise ValueError(f"pred_type inconnu : {pred_type}")
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device
        self.pred_type = pred_type

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """x0 : (B, 1, L) fenêtres NORMALISÉES."""
        B = x0.shape[0]
        t = torch.randint(0, self.schedule.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        if self.pred_type == "eps":
            target = noise
        else:
            ab = self.schedule.alpha_bars[t].view(-1, 1, 1)
            target = ab.sqrt() * noise - (1.0 - ab).sqrt() * x0
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(
        self,
        n       : int,
        length  : int,
        seed    : Optional[int] = None,
        batch   : int = 250,
        clip_x0 : float = 15.0,
    ) -> np.ndarray:
        """
        Sampling ancestral complet (T pas), paramétré par x̂₀ avec clipping
        (le « clip_denoised » standard) : x̂₀ = (x_t − √(1-ᾱ_t)·ε̂)/√ᾱ_t est
        borné à ±clip_x0 avant de recalculer la moyenne posterior. Garde-fou
        contre la divergence (sans lui, un ε̂ mal calibré s'amplifie en 1/√ᾱ) ;
        neutre pour un modèle entraîné : |z| réel max ≈ 12 < 15.

        Renvoie (n, length) en espace NORMALISÉ — la dénormalisation
        appartient à l'appelant (norm.json).
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.model.eval()
        sched = self.schedule
        out = []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            x = torch.randn(b, 1, length, device=self.device)
            for t in reversed(range(sched.T)):
                tt = torch.full((b,), t, device=self.device, dtype=torch.long)
                pred = self.model(x, tt)
                ab = sched.alpha_bars[t]
                abp = sched.alpha_bars_prev[t]
                alpha = sched.alphas[t]
                beta = sched.betas[t]

                if self.pred_type == "eps":
                    x0 = (x - (1.0 - ab).sqrt() * pred) / ab.sqrt()
                else:   # v : x̂₀ = √ᾱ·x_t − √(1-ᾱ)·v̂
                    x0 = ab.sqrt() * x - (1.0 - ab).sqrt() * pred
                x0 = x0.clamp(-clip_x0, clip_x0)
                mean = (abp.sqrt() * beta / (1.0 - ab)) * x0 \
                     + (alpha.sqrt() * (1.0 - abp) / (1.0 - ab)) * x
                if t > 0:
                    x = mean + sched.posterior_var[t].sqrt() * torch.randn_like(x)
                else:
                    x = mean
            out.append(x.squeeze(1).float().cpu().numpy())
        return np.concatenate(out, axis=0)


# ============================================================
# PERSISTANCE
# ============================================================
def save_ddpm(out_dir: str, model: UNet1D, ema: EMA, config: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"model": {k: v.cpu() for k, v in model.state_dict().items()},
         "ema": ema.state_dict(),
         "config": config},
        os.path.join(out_dir, "checkpoint.pt"),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_ddpm(model_dir: str, device: str = "cpu",
              use_ema: bool = True) -> Tuple[DDPM, dict]:
    """Recharge modèle + schedule depuis un dossier d'expérience."""
    ckpt = torch.load(os.path.join(model_dir, "checkpoint.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = UNet1D(channels=tuple(cfg["channels"]), t_dim=cfg["t_dim"])
    model.load_state_dict(ckpt["model"])
    if use_ema:
        ema = EMA(model, decay=cfg.get("ema_decay", 0.999))
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
    model = model.to(device)
    schedule = DiffusionSchedule(T=cfg["T"], kind=cfg["schedule"], device=device)
    return DDPM(model, schedule, device=device,
                pred_type=cfg.get("pred_type", "eps")), cfg

# train_diffusion.py
"""
Entraînement du DDPM v1 sur les fenêtres de rendements du split TRAIN RL.

Convention repo : l'expérience vit dans SON dossier (models/diffusion_v1/) ;
la prédiction est écrite dans le rapport AVANT le run ; le modèle n'est jugé
que par validate_diffusion.py contre le protocole pré-enregistré.

Sorties : models/diffusion_v1/{checkpoint.pt, config.json, norm.json, loss_curve.png}
"""
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from diffusion.dataset import (
    WindowConfig, load_train_windows, compute_norm, normalize, save_norm,
)
from diffusion.model import UNet1D
from diffusion.schedule import DiffusionSchedule
from diffusion.ddpm import DDPM, EMA, save_ddpm


@dataclass
class DiffusionTrainConfig:
    # ── Processus de diffusion ────────────────────────────────────────
    T          : int   = 1000
    schedule   : str   = "cosine"

    # [TECHNIQUE] 'eps' = prédire le bruit (Ho 2020) | 'v' = v-prediction
    # (Salimans & Ho 2022). v est mieux conditionnée sur des rendements quasi
    # blancs : l'ε-pred de la v1 rétrécissait la variance échantillonnée de
    # 45 % (biais minuscule composé sur T pas) — mesuré, pas supposé.
    pred_type  : str   = "eps"

    # ── Réseau ────────────────────────────────────────────────────────
    channels   : Tuple[int, ...] = (32, 64, 128)
    t_dim      : int   = 128

    # ── Optimisation ──────────────────────────────────────────────────
    lr         : float = 2e-4
    # [TECHNIQUE] LR final de la décroissance cosine. La calibration fine
    # d'échelle se joue en fin d'entraînement : un LR constant y laisse les
    # poids « vibrer » autour de l'optimum.
    lr_min     : float = 1e-5
    batch      : int   = 128
    steps      : int   = 25_000
    ema_decay  : float = 0.999

    # ── Reproductibilité / sorties ────────────────────────────────────
    seed       : int   = 42
    out_dir    : str   = "models/diffusion_v1"
    log_every  : int   = 500


def main(cfg: DiffusionTrainConfig = DiffusionTrainConfig()):
    from train import set_all_seeds          # convention seeds du repo
    set_all_seeds(cfg.seed)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥  Device : {device}")

    # ---- Données ----
    win_cfg = WindowConfig()
    windows, meta = load_train_windows(win_cfg)
    mu, sigma = compute_norm(windows)
    print(f"📦 {len(windows)} fenêtres de {win_cfg.window} j | "
          f"μ={mu:.2e} σ={sigma:.4f}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_norm(os.path.join(cfg.out_dir, "norm.json"), mu, sigma, win_cfg)

    data = torch.from_numpy(normalize(windows, mu, sigma)).float().to(device)

    # ---- Modèle ----
    model = UNet1D(channels=cfg.channels, t_dim=cfg.t_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 UNet1D {cfg.channels} — {n_params/1e6:.2f}M paramètres")

    schedule = DiffusionSchedule(T=cfg.T, kind=cfg.schedule, device=device)
    ddpm = DDPM(model, schedule, device=device, pred_type=cfg.pred_type)
    ema = EMA(ddpm.model, decay=cfg.ema_decay)
    opt = torch.optim.AdamW(ddpm.model.parameters(), lr=cfg.lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.steps, eta_min=cfg.lr_min)

    # ---- Boucle ----
    t0 = time.time()
    losses, log_steps, log_losses = [], [], []
    n = len(data)
    ddpm.model.train()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, n, (cfg.batch,), device=device)
        x0 = data[idx].unsqueeze(1)
        loss = ddpm.loss(x0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_sched.step()
        ema.update(ddpm.model)
        losses.append(float(loss.detach()))

        if step % cfg.log_every == 0:
            avg = float(np.mean(losses[-cfg.log_every:]))
            log_steps.append(step)
            log_losses.append(avg)
            eta = (time.time() - t0) / step * (cfg.steps - step)
            print(f"   step {step:>6}/{cfg.steps} | loss {avg:.4f} "
                  f"| ETA {eta/60:.0f} min")

    duree = time.time() - t0
    print(f"⏱  Entraînement : {duree/60:.1f} min")

    # ---- Sauvegarde ----
    config = {
        **asdict(cfg),
        "channels"   : list(cfg.channels),
        "window"     : win_cfg.window,
        "stride"     : win_cfg.stride,
        "n_windows"  : int(len(windows)),
        "n_params"   : int(n_params),
        "mu"         : mu,
        "sigma"      : sigma,
        "final_loss" : float(np.mean(losses[-cfg.log_every:])),
        "duree_min"  : round(duree / 60, 1),
        "device"     : device,
    }
    save_ddpm(cfg.out_dir, ddpm.model, ema, config)
    print(f"💾 Checkpoint : {cfg.out_dir}/checkpoint.pt")

    # ---- Courbe de loss ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log_steps, log_losses, lw=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel(f"loss MSE (moyenne / {cfg.log_every} steps)")
    ax.set_title(f"DDPM v1 — {cfg.schedule}, T={cfg.T}, "
                 f"{n_params/1e6:.1f}M params, {device}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "loss_curve.png"), dpi=130)
    print(f"📈 Courbe : {cfg.out_dir}/loss_curve.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-type", choices=["eps", "v"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = DiffusionTrainConfig()
    if args.pred_type:
        cfg.pred_type = args.pred_type
    if args.steps:
        cfg.steps = args.steps
    if args.out_dir:
        cfg.out_dir = args.out_dir
    main(cfg)

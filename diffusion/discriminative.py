# diffusion/discriminative.py
"""
Score discriminatif façon TimeGAN : un petit classifieur essaie de distinguer
fenêtres réelles et synthétiques. acc ≈ 0.5 = indiscernables.

Le classifieur est FIGÉ À L'AVANCE (architecture + hyperparamètres) : le seuil
É5 n'a de sens que si le juge ne change pas entre les générateurs. Un
discriminateur plus gros trouverait toujours quelque chose ; celui-ci est
volontairement modeste et identique pour tous.

Trois leçons de la calibration v1, intégrées ici :
- SPLIT PAR BLOCS ALÉATOIRES PURGÉS (embargo façon de Prado) : les fenêtres
  réelles se chevauchent (stride 1) ; un split aléatoire fuit, et un split
  « derniers 20 % en test » fait apprendre l'ÉPOQUE au lieu du réalisme
  (sanité entrelacée à 0.39 au lieu de 0.5). Blocs contigus tirés au hasard
  → le test couvre toutes les périodes ; embargo → fuite bornée à un
  demi-recouvrement.
- CLIP DES ENTRÉES à ±8σ : les rendements extrêmes (|z| jusqu'à 12) saturent
  le GRU et provoquaient des collapses d'entraînement (acc bloquée à 0.5).
- MÉDIANE DE 3 SEEDS : le juge v1 était bimodal (0.88 / 0.94 / 0.50 selon le
  seed) ; la médiane de 3 entraînements stabilise la métrique.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DiscConfig:
    # Hyperparamètres FIGÉS du juge.
    hidden    : int   = 64
    lr        : float = 1e-3
    epochs    : int   = 30
    batch     : int   = 128
    test_frac : float = 0.2
    n_seeds   : int   = 3       # médiane de n_seeds entraînements
    clip      : float = 8.0     # clip des entrées (en σ)
    block     : int   = 256     # taille des blocs contigus du split (fenêtres)
    embargo   : int   = 128     # fenêtres écartées autour des blocs de test
    seed      : int   = 0


class GRUClassifier(nn.Module):
    """
    GRU(2→hidden) sur (r, |r|), sorties moyennées dans le temps → logit.
    Donner |r| en entrée rend le juge directement sensible à la dynamique de
    volatilité — exactement ce qu'on veut qu'il sache détecter (clustering).
    ~14k paramètres.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([x, x.abs()], dim=-1)
        out, _ = self.gru(feats)
        return self.head(out.mean(dim=1)).squeeze(-1)


def _block_split(
    n         : int,
    groups    : Optional[np.ndarray],
    test_frac : float,
    rng       : np.random.Generator,
    block     : int = 256,
    embargo   : int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices (train, test).

    groups=None (samples i.i.d., ex : synthétique) → split aléatoire simple.
    groups fournis (fenêtres réelles, ordonnées dans le temps au sein de
    chaque groupe) → blocs contigus de `block` fenêtres ; ~test_frac des blocs
    tirés au hasard en test ; les fenêtres de train à moins de `embargo`
    positions d'un bloc de test sont ÉCARTÉES (purge).
    """
    if groups is None:
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * test_frac)))
        return perm[n_test:], perm[:n_test]

    groups = np.asarray(groups)
    train_idx, test_idx = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)          # ordre temporel préservé
        pos = np.arange(len(idx))
        block_id = pos // block
        n_blocks = block_id.max() + 1
        n_test_blocks = max(1, int(round(n_blocks * test_frac)))
        test_blocks = rng.choice(n_blocks, n_test_blocks, replace=False)

        is_test = np.isin(block_id, test_blocks)
        # purge : distance (en positions) au bloc de test le plus proche
        test_pos = pos[is_test]
        if len(test_pos) and len(test_pos) < len(pos):
            dist = np.abs(pos[:, None] - test_pos[None, :]).min(axis=1)
            keep_train = (~is_test) & (dist > embargo)
        else:
            keep_train = ~is_test
        train_idx.append(idx[keep_train])
        test_idx.append(idx[is_test])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def _train_once(
    x_train, y_train, x_test, y_test, cfg: DiscConfig, seed: int, device: str,
) -> float:
    torch.manual_seed(seed)
    model = GRUClassifier(cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(x_train)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(x_test), cfg.batch):
            logits = model(x_test[i:i + cfg.batch].to(device))
            preds.append((logits > 0).float().cpu())
        preds = torch.cat(preds)
    return float((preds == y_test).float().mean())


def discriminative_score(
    real         : np.ndarray,
    synth        : np.ndarray,
    real_groups  : Optional[np.ndarray] = None,
    synth_groups : Optional[np.ndarray] = None,
    cfg          : DiscConfig = DiscConfig(),
    device       : str = "cpu",
) -> dict:
    """
    Entraîne le juge figé réel (label 1) vs synthétique (label 0) sur
    cfg.n_seeds seeds, renvoie la MÉDIANE des accuracies test (classes
    équilibrées). synth_groups sert aux comparaisons réel-vs-réel (les deux
    camps reçoivent alors le même split par blocs purgés — symétrie).

    ⚠️ real et synth doivent être NORMALISÉS avec le MÊME μ/σ (celui du réel),
    sinon le classifieur gagne trivialement sur l'échelle.
    """
    rng = np.random.default_rng(cfg.seed)

    r_train, r_test = _block_split(len(real), real_groups, cfg.test_frac,
                                   rng, cfg.block, cfg.embargo)
    s_train, s_test = _block_split(len(synth), synth_groups, cfg.test_frac,
                                   rng, cfg.block, cfg.embargo)

    def balance(a, b):
        m = min(len(a), len(b))
        return (rng.choice(a, m, replace=False),
                rng.choice(b, m, replace=False))

    r_train, s_train = balance(r_train, s_train)
    r_test,  s_test  = balance(r_test,  s_test)

    def make_xy(r_idx, s_idx):
        x = np.concatenate([real[r_idx], synth[s_idx]]).astype(np.float32)
        x = np.clip(x, -cfg.clip, cfg.clip)
        y = np.concatenate([np.ones(len(r_idx)), np.zeros(len(s_idx))]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    x_train, y_train = make_xy(r_train, s_train)
    x_test,  y_test  = make_xy(r_test,  s_test)

    accs = sorted(
        _train_once(x_train, y_train, x_test, y_test, cfg,
                    seed=cfg.seed + k, device=device)
        for k in range(cfg.n_seeds)
    )
    return {
        'acc'      : float(np.median(accs)),
        'acc_runs' : [float(a) for a in accs],
        'n_train'  : int(len(x_train)),
        'n_test'   : int(len(x_test)),
    }

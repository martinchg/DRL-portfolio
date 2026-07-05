# test_diffusion.py — Phase 1 diffusion : dataset, protocole de validation, DDPM.
#
# Credo validation-first : le PROTOCOLE est testé sur des distributions aux
# propriétés CONNUES (bruit blanc, GARCH simulé, Student-t) AVANT de juger le
# moindre modèle. Un protocole faux rend tout le reste faux — même logique que
# les tests de l'environnement RL.
#
# Tous les tests : rapides, sans réseau, données synthétiques.
import numpy as np
import pandas as pd
import pytest
import torch

from diffusion.dataset import (
    WindowConfig, extract_windows, segment_returns, log_returns_from_price,
    compute_norm, normalize, denormalize, save_norm, load_norm,
)
from diffusion.metrics import (
    ValidationCriteria, acf, pooled_moments, window_stats,
    nn_distances_cross, nn_distances_loo, sampling_band,
    build_real_bands, generator_summary, judge_criteria,
)
from diffusion.baselines import (
    sample_gaussian_iid, sample_bootstrap_iid, simulate_garch,
)
from diffusion.discriminative import DiscConfig, discriminative_score, _block_split


RNG = lambda s=0: np.random.default_rng(s)


# ============================================================
# DATASET
# ============================================================
def _price_df_two_segments(n=400):
    """2 segments de prix avec un SAUT énorme à la jonction (GOOGL→SPY-like)."""
    rng = RNG(1)
    p1 = 1000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    p2 = 300.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({
        'price'      : np.concatenate([p1, p2]),
        'segment_id' : np.repeat([0, 1], n),
        'ticker'     : np.repeat(['AAA', 'BBB'], n),
    })
    return df


def test_log_returns_from_price_exact():
    prices = np.array([100.0, 110.0, 99.0])
    r = log_returns_from_price(prices)
    assert np.allclose(r, [np.log(1.10), np.log(99 / 110)])


def test_windows_ne_traversent_pas_les_segments():
    # LE piège : une fenêtre à cheval verrait un faux krach de -70 %.
    df = _price_df_two_segments(n=400)
    cfg = WindowConfig(window=64, stride=1)
    windows, meta = extract_windows(df, cfg)

    # Nombre exact : (400-1 rendements - 64 + 1) par segment
    assert len(windows) == 2 * (399 - 64 + 1)
    assert windows.shape[1] == 64

    # Aucun rendement de la taille du saut de jonction (|log(300/1000)| ≈ 1.2)
    assert np.abs(windows).max() < 0.5

    # meta cohérente
    assert set(meta['segment_id'].unique()) == {0, 1}
    assert (meta.groupby('segment_id')['start'].min() == 0).all()


def test_windows_stride():
    df = _price_df_two_segments(n=200)
    w1, _ = extract_windows(df, WindowConfig(window=64, stride=1))
    w5, _ = extract_windows(df, WindowConfig(window=64, stride=5))
    assert len(w5) < len(w1)
    # les fenêtres stride=5 sont un sous-ensemble des stride=1
    assert np.allclose(w5[0], w1[0])
    assert np.allclose(w5[1], w1[5])


def test_normalisation_aller_retour(tmp_path):
    w = RNG(2).normal(0.0005, 0.02, (50, 32)).astype(np.float32)
    mu, sigma = compute_norm(w)
    z = normalize(w, mu, sigma)
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1.0) < 1e-6
    assert np.allclose(denormalize(z, mu, sigma), w, atol=1e-7)

    path = str(tmp_path / "norm.json")
    save_norm(path, mu, sigma, WindowConfig(window=32, stride=1))
    mu2, sigma2, cfg2 = load_norm(path)
    assert (mu2, sigma2) == (mu, sigma) and cfg2.window == 32


def test_segment_returns_sans_colonne_segment():
    df = pd.DataFrame({'price': [100.0, 101.0, 102.0]})
    segs = segment_returns(df)
    assert list(segs.keys()) == [0] and len(segs[0]) == 2


# ============================================================
# MÉTRIQUES — testées sur des distributions à propriétés connues
# ============================================================
def test_acf_bruit_blanc_proche_de_zero():
    w = RNG(3).standard_normal((400, 128))
    a = acf(w, max_lag=10)
    assert np.abs(a).max() < 0.05          # ≈ 0 à l'erreur d'échantillonnage près


def test_acf_ar1_retrouve_le_coefficient():
    # AR(1) : ACF(k) = φ^k — vérité analytique.
    phi, n, L = 0.6, 500, 128
    rng = RNG(4)
    eps = rng.standard_normal((n, L + 100))
    x = np.zeros_like(eps)
    for t in range(1, eps.shape[1]):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    a = acf(x[:, 100:], max_lag=3)
    # estimateur biaisé par fenêtre → tolérance large mais ordonnée
    assert abs(a[0] - phi) < 0.08
    assert abs(a[1] - phi ** 2) < 0.08
    assert a[0] > a[1] > a[2]


def test_garch_a_du_clustering_pas_la_gaussienne():
    # Le cœur d'É4 : ACF(|r|) > 0 pour GARCH, ≈ 0 pour i.i.d. gaussien.
    g = simulate_garch(300, 256, omega=4e-6, alpha=0.10, beta=0.88, rng=RNG(5))
    iid = sample_gaussian_iid(300, 256, 0.0, 0.02, rng=RNG(6))
    acf_g = acf(np.abs(g), max_lag=10)
    acf_i = acf(np.abs(iid), max_lag=10)
    assert acf_g[0] > 0.10                  # clustering net au lag 1
    assert acf_g[:5].min() > 0.03           # persistance
    assert np.abs(acf_i).max() < 0.05       # rien pour l'i.i.d.


def test_kurtosis_student_superieure_gaussienne():
    rng = RNG(7)
    gauss = rng.standard_normal((200, 256))
    student = rng.standard_t(4, size=(200, 256))   # kurtosis excès théorique = 3
    k_g = pooled_moments(gauss)['kurtosis_excess']
    k_s = pooled_moments(student)['kurtosis_excess']
    assert abs(k_g) < 0.3
    assert k_s > 1.5


def test_window_stats_valeurs_exactes():
    # Fenêtre déterministe : +1 % puis -2 % → drawdown connu.
    r = np.array([[0.01, -0.02, 0.0, 0.0]])
    ws = window_stats(r)
    p2 = np.exp(0.01 - 0.02)
    assert np.isclose(ws['terminal'][0], -0.01)
    assert np.isclose(ws['max_dd'][0], 1.0 - p2 / np.exp(0.01))


def test_nn_cross_retrouve_la_copie():
    pool = RNG(8).standard_normal((200, 64))
    queries = pool[[3, 50]] + 1e-4          # quasi-copies
    d = nn_distances_cross(queries, pool)
    assert d.max() < 1e-2


def test_nn_loo_exclut_les_voisins_chevauchants():
    # Fenêtres glissantes stride 1 d'une même série : sans exclusion, la
    # distance NN serait celle de la fenêtre décalée d'un jour (minuscule).
    series = RNG(9).standard_normal(400)
    L = 64
    windows = np.stack([series[i:i + L] for i in range(400 - L + 1)])
    meta = pd.DataFrame({
        'segment_id': np.zeros(len(windows), dtype=int),
        'start': np.arange(len(windows)),
    })
    d_loo = nn_distances_loo(windows, meta, window_len=L)

    # distance naïve (sans exclusion) au plus proche = fenêtre décalée d'1 jour
    d_naive = nn_distances_cross(windows[:5], np.delete(windows, [0, 1, 2, 3, 4], axis=0))
    assert d_loo.min() > 1.0                # bruit blanc : fenêtres disjointes loin
    assert d_naive.min() < d_loo[:5].min()  # la naïve trouve le voisin décalé


def test_sampling_band_contient_la_vraie_valeur():
    w = RNG(10).standard_normal((2000, 64))
    band = sampling_band(lambda x: pooled_moments(x)['kurtosis_excess'],
                         w, n_draws=50, sample_size=500, rng=RNG(11))
    assert band['lo'][0] <= 0.0 <= band['hi'][0]    # kurtosis excès vraie = 0


def test_judge_criteria_garch_passe_gaussienne_echoue():
    # Mini bout-en-bout du protocole sur données 100 % synthétiques :
    # le "réel" est un GARCH-t (queues + clustering), B0 gaussien doit échouer.
    real = simulate_garch(1500, 128, omega=4e-6, alpha=0.10, beta=0.88,
                          nu=5.0, rng=RNG(12))
    crit = ValidationCriteria(max_lag=10, acf_r_lags=(1, 2, 3),
                              n_eval_windows=400, n_band_draws=40)
    bands = build_real_bands(real, crit, rng=RNG(13))

    # réel-vs-réel (É1) : un tirage de fenêtres réelles passe É2-É4
    sub = real[RNG(14).choice(len(real), 400, replace=False)]
    verdict_real = judge_criteria(generator_summary(sub, crit.max_lag), bands, crit)
    assert verdict_real['E2_queues']['pass']
    assert verdict_real['E3_acf_parasite']['pass']
    assert verdict_real['E4_clustering']['pass']

    # B0 gaussienne : échoue queues ET clustering
    b0 = sample_gaussian_iid(400, 128, 0.0, float(real.std()), rng=RNG(15))
    verdict_b0 = judge_criteria(generator_summary(b0, crit.max_lag), bands, crit)
    assert not verdict_b0['E2_queues']['pass']
    assert not verdict_b0['E4_clustering']['pass']
    assert not verdict_b0['all_pass']

    # B1 bootstrap : marginales OK (É2) mais clustering détruit (É4)
    b1 = sample_bootstrap_iid(400, 128, real.ravel(), rng=RNG(16))
    verdict_b1 = judge_criteria(generator_summary(b1, crit.max_lag), bands, crit)
    assert verdict_b1['E2_queues']['pass']
    assert not verdict_b1['E4_clustering']['pass']


def test_judge_criteria_e5_e6():
    crit = ValidationCriteria()
    summary = {'moments': {'kurtosis_excess': 3.0},
               'acf_r': [0.0] * 20, 'acf_absr': [0.2] + [0.1] * 19,
               'acf_absr_sum': 2.0}
    bands = {
        'kurtosis_excess': {'lo': [1.5], 'hi': [6.0], 'median': [3.0]},
        'acf_r': {'lo': [-0.05] * 20, 'hi': [0.05] * 20, 'median': [0.0] * 20},
        'acf_absr': {'lo': [0.1] * 20, 'hi': [0.3] * 20, 'median': [0.2] * 20},
        'acf_absr_sum': {'lo': [1.0], 'hi': [3.0], 'median': [2.0]},
    }
    v = judge_criteria(summary, bands, crit,
                       disc_acc=0.62, garch_disc_acc=0.60,
                       nn_median=5.0, nn_real_ref=4.0)
    assert v['E5_discriminatif']['pass']        # 0.62 <= 0.60 + 0.05
    assert v['E6_memorisation']['pass']         # 5.0 >= 4.0
    v2 = judge_criteria(summary, bands, crit,
                        disc_acc=0.70, garch_disc_acc=0.60,
                        nn_median=3.0, nn_real_ref=4.0)
    assert not v2['E5_discriminatif']['pass']
    assert not v2['E6_memorisation']['pass']


# ============================================================
# BASELINE GARCH FITTÉE (arch) — fit sur données simulées connues
# ============================================================
def test_fit_garch_retrouve_le_clustering():
    from diffusion.baselines import fit_garch_per_segment, sample_garch_fitted
    # Série GARCH simulée aux paramètres connus → le fit doit produire un
    # générateur qui a lui-même du clustering (pas d'égalité paramétrique
    # exigée : le MLE sur 3000 points reste bruité).
    series = simulate_garch(1, 3000, omega=4e-6, alpha=0.10, beta=0.88,
                            nu=6.0, rng=RNG(17))[0].astype(np.float64)
    fitted = fit_garch_per_segment({0: series})
    p = fitted[0]
    assert 0.0 < p['alpha'] < 0.5 and 0.5 < p['beta'] < 1.0
    assert p['alpha'] + p['beta'] < 1.0

    sim = sample_garch_fitted(200, 128, fitted, rng=RNG(18))
    assert sim.shape == (200, 128)
    assert acf(np.abs(sim), 5)[0] > 0.05        # le clustering survit au fit


# ============================================================
# DISCRIMINATIF
# ============================================================
def test_block_split_purge_et_couverture():
    groups = np.repeat([0, 1], 1000)
    train, test = _block_split(2000, groups, 0.2, RNG(19), block=100, embargo=50)

    assert len(np.intersect1d(train, test)) == 0
    for g in [0, 1]:
        lo, hi = g * 1000, (g + 1) * 1000
        tr = train[(train >= lo) & (train < hi)] - lo
        te = test[(test >= lo) & (test < hi)] - lo
        # ~20 % des blocs en test dans CHAQUE groupe
        assert 100 <= len(te) <= 400
        # purge : aucun index de train à moins d'`embargo` positions d'un test
        assert np.abs(tr[:, None] - te[None, :]).min() > 50


def test_discriminative_separe_ce_qui_est_separable():
    # Config du juge FIGÉE (DiscConfig par défaut), données réduites (L=32)
    # pour rester dans la suite rapide.
    rng = RNG(20)
    a = rng.standard_normal((200, 32)).astype(np.float32)
    b = (rng.standard_normal((200, 32)) * 2.0).astype(np.float32)   # vol ×2
    score = discriminative_score(a, b, cfg=DiscConfig())
    assert score['acc'] > 0.8


def test_discriminative_ne_separe_pas_l_identique():
    rng = RNG(21)
    a = rng.standard_normal((200, 32)).astype(np.float32)
    b = rng.standard_normal((200, 32)).astype(np.float32)
    score = discriminative_score(a, b, cfg=DiscConfig())
    assert abs(score['acc'] - 0.5) < 0.12


# ============================================================
# DDPM : schedule, U-Net, EMA, entraînement minimal
# ============================================================
from diffusion.schedule import DiffusionSchedule, make_betas
from diffusion.model import UNet1D
from diffusion.ddpm import DDPM, EMA, save_ddpm, load_ddpm


def test_schedule_invariants():
    for kind in ['linear', 'cosine']:
        s = DiffusionSchedule(T=100, kind=kind)
        ab = s.alpha_bars
        assert torch.all(ab[1:] < ab[:-1])          # strictement décroissant
        assert 0.0 < float(ab[-1]) < float(ab[0]) <= 1.0
        assert torch.all(s.betas > 0) and torch.all(s.betas <= 0.999)
        assert torch.all(s.posterior_var >= 0)


def test_q_sample_debut_et_fin():
    s = DiffusionSchedule(T=200, kind='cosine')
    torch.manual_seed(0)
    x0 = torch.randn(500, 1, 64)
    noise = torch.randn_like(x0)

    # t=0 : quasi pas de bruit
    x_t0 = s.q_sample(x0, torch.zeros(500, dtype=torch.long), noise)
    assert float((x_t0 - x0).abs().mean()) < 0.15

    # t=T-1 : ≈ N(0,1), signal détruit
    t_max = torch.full((500,), 199, dtype=torch.long)
    x_T = s.q_sample(x0, t_max, noise)
    assert abs(float(x_T.mean())) < 0.02
    assert abs(float(x_T.std()) - 1.0) < 0.02
    corr = float((x_T * x0).mean() / (x_T.std() * x0.std()))
    assert abs(corr) < 0.05                          # décorrélé de x0


def test_unet_shape_et_init_zero():
    m = UNet1D(channels=(8, 16, 32), t_dim=32)
    x = torch.randn(4, 1, 64)
    t = torch.randint(0, 10, (4,))
    y = m(x, t)
    assert y.shape == x.shape
    # conv finale zero-init → le réseau prédit exactement 0 au départ
    assert torch.allclose(y, torch.zeros_like(y))
    # les gradients atteignent la conv de sortie
    y.sum().backward()
    assert m.out_conv.weight.grad is not None


def test_ddpm_overfit_mini():
    torch.manual_seed(0)
    model = UNet1D(channels=(8, 16, 32), t_dim=32)
    sched = DiffusionSchedule(T=50, kind='cosine')
    ddpm = DDPM(model, sched, device='cpu')
    tt = torch.linspace(0, 4 * np.pi, 64)
    x0 = torch.sin(tt).repeat(8, 1, 1)               # 8 fenêtres identiques

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(150):
        loss = ddpm.loss(x0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    debut, fin = np.mean(losses[:20]), np.mean(losses[-20:])
    assert fin < debut * 0.7                          # la loss descend nettement


@pytest.mark.parametrize("pred_type", ["eps", "v"])
def test_ddpm_sample_shape_et_valeurs(pred_type):
    torch.manual_seed(0)
    model = UNet1D(channels=(8, 16, 32), t_dim=32)
    sched = DiffusionSchedule(T=20, kind='cosine')
    ddpm = DDPM(model, sched, device='cpu', pred_type=pred_type)
    loss = ddpm.loss(torch.randn(4, 1, 64))
    assert torch.isfinite(loss)
    out = ddpm.sample(5, 64, seed=1)
    assert out.shape == (5, 64)
    assert np.all(np.isfinite(out))
    # le clip de x̂₀ borne la dynamique même pour un modèle non entraîné
    assert float(np.abs(out).max()) < 30.0


@pytest.mark.parametrize("pred_type", ["eps", "v"])
def test_sampler_oracle_reproduit_la_gaussienne(pred_type):
    # Test décisif écrit pour disculper le sampler après le NO-GO v2 : avec le
    # débruiteur OPTIMAL analytique pour des données N(0,1) (E[ε|x_t] =
    # √(1-ᾱ)·x_t ; E[v|x_t] = 0), la boucle ancestrale doit reproduire N(0,1).
    # Un écart de std ici = bug de sampling, pas d'apprentissage.
    sched = DiffusionSchedule(T=200, kind='cosine')

    class Oracle:
        def eval(self): pass
        def __call__(self, x, t):
            if pred_type == "eps":
                return (1 - sched.alpha_bars[t[0]]).sqrt() * x
            return torch.zeros_like(x)

    ddpm = DDPM.__new__(DDPM)
    ddpm.model = Oracle()
    ddpm.schedule = sched
    ddpm.device = 'cpu'
    ddpm.pred_type = pred_type
    out = DDPM.sample(ddpm, 300, 64, seed=3)
    assert abs(float(out.std()) - 1.0) < 0.03
    assert abs(float(out.mean())) < 0.03


def test_ema_converge_et_copie():
    m = UNet1D(channels=(8, 16, 32), t_dim=32)
    ema = EMA(m, decay=0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    for _ in range(30):                               # 0.5^30 ≈ 1e-9
        ema.update(m)
    ema.copy_to(m)
    ref = {k: p.detach().clone() for k, p in m.named_parameters()}
    for k, v in ema.shadow.items():
        assert torch.allclose(ref[k], v, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = UNet1D(channels=(8, 16, 32), t_dim=32)
    sched = DiffusionSchedule(T=20, kind='cosine')
    ema = EMA(model, decay=0.9)
    for _ in range(3):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01)
        ema.update(model)
    cfg = {"T": 20, "schedule": "cosine", "channels": [8, 16, 32],
           "t_dim": 32, "ema_decay": 0.9}
    save_ddpm(str(tmp_path), model, ema, cfg)

    ddpm, cfg2 = load_ddpm(str(tmp_path), device='cpu', use_ema=True)
    assert cfg2["T"] == 20
    # les poids chargés sont les poids EMA
    for k, p in ddpm.model.named_parameters():
        assert torch.allclose(p, ema.shadow[k], atol=1e-6)
    out = ddpm.sample(2, 64, seed=3)
    assert out.shape == (2, 64) and np.all(np.isfinite(out))

# DRL Portfolio — instructions projet

Projet perso de Martin (candidatures stage Quant) : agent de trading PPO (Stable-Baselines3)
sur données journalières réelles, comparé au Buy & Hold. Langue de travail : français.

## ⚠️ Règle n°0 : le rapport .tex est le livrable prioritaire

`reports/rapport_drl_portfolio.tex` est le support d'apprentissage de Martin : il doit
pouvoir l'**apprendre par cœur** et raconter le projet en entretien comme SON cheminement.
À chaque grosse tâche, la mise à jour du rapport fait partie de la tâche (pas une option) :

- **Écrire le raisonnement financier/méthodologique, jamais les détails de code.**
  Les revues de code n'ont pas leur place dans le rapport ; un bug n'y figure que si
  sa conséquence financière s'explique (ex : « la validation contenait de faux krachs
  → le modèle sélectionné était le mauvais »).
- **Format de chaque étape du récit** : symptôme observé → diagnostic → interprétation
  financière (pourquoi c'est grave pour un backtest / ce qu'en dirait un desk) →
  décision prise → résultat mesuré → ce que ça ouvre ensuite. Les transitions entre
  étapes doivent paraître naturelles, presque inévitables.
- **Première personne** (« j'ai constaté… j'en ai déduit… je suis donc passé à… ») :
  c'est le cheminement mental de Martin, pas un changelog d'outil. Il assume avoir
  utilisé Claude ; ce qui compte est qu'il possède le raisonnement.
- **Tout est documenté avec ses chiffres** : chaque expérience, chaque prédiction faite
  avant un run (et si elle s'est réalisée ou non), chaque protocole de test.
- Le rapport garde une **base théorique en ouverture** (MDP, PPO, single-agent vs
  multi-agent) avant le récit chronologique.
- **Structure éditoriale — sections figées vs vivantes.** Le récit chronologique
  (actes 1-2 : construction/fiabilisation, puis walk-forward/stress) est FIGÉ : on n'y
  ajoute pas une itération par tâche, les anciens chiffres y restent comme jalons.
  Les sections VIVANTES (Résultats, Limites, Glossaire) se mettent à jour en écrasant.
  Un nouveau chapitre narratif n'est créé que quand une QUESTION est fermée (réponse
  mesurée), en regroupant les expériences qui y répondent — ex. futur « Acte 3 :
  attaquer la dépendance de régime » = multi-seeds + features de régime + position
  continue en sous-sections. Les micro-changements techniques ne reçoivent pas de
  section : mise à jour des sections vivantes + une ligne au journal des révisions.
- Après édition : recompiler avec `tectonic` et vérifier le rendu des figures.

## ⚠️ Règle de maintenance des livrables (à chaque grosse tâche)

Après toute grosse tâche (changement d'environnement, de reward, de features, de protocole
d'évaluation, réentraînement, nouveaux résultats…), **vérifier que les 3 livrables de
présentation sont à jour** et les régénérer/adapter si besoin :

1. **Assets + dashboard statique** : `python reports/build_assets.py`
   → régénère `reports/figures/*.png` (figures de RÉSULTATS), `reports/metrics.json` et
   `docs/index.html` (page GitHub Pages). Si le contenu narratif du HTML (sections
   "ce qui fonctionne / ce qui reste fragile", KPI cards) n'est plus exact, éditer le
   template dans `build_assets.py`, pas le HTML généré.
   → Figures PÉDAGOGIQUES (schémas MDP/pipeline/walk-forward, OU, clipping PPO,
   dispersion seeds, volatility clustering, frontière efficiente) : script séparé
   `python reports/build_concept_figures.py` (autonome, pas de modèle requis).
2. **Rapport LaTeX** : `reports/rapport_drl_portfolio.tex` — mettre à jour les chiffres et
   sections impactés (chronologie §6, résultats §7, limites §9 ; les chiffres sont aussi
   dans `reports/metrics.json`), puis recompiler : `tectonic reports/rapport_drl_portfolio.tex`
   (tectonic est installé). Le PDF compilé est tracké par git (exception dans .gitignore).
3. **Dashboard Streamlit** : vérifier que `dashboard.py` reflète les changements
   (chemins des modèles, métriques, config env) ; smoke test :
   `python -m pytest tests/test_dashboard.py`.

Si les chiffres changent, mettre aussi à jour la section **Results** du `README.md`.

## Commandes

- Tests : `.venv/bin/python -m pytest` — 67 tests ; `-m "not slow and not network"` pour
  les rapides (~7 s). Toujours lancer avant de conclure une tâche.
- Entraînement complet : `python train.py` (~10-15 min, écrase `models/ppo_single|ppo_multi`)
- Évaluation : `python evaluate.py` (full-split déterministe + robustesse + cross-ticker)
- Walk-forward : `python walk_forward.py` (~25 min, 5 folds ; écrit `reports/walk_forward.json`
  que `build_assets.py` intègre automatiquement aux figures et au dashboard)
- Robustesse au seed : `python seed_robustness.py` (~35 min, 4 réentraînements Multi
  → `reports/seed_robustness.json` = la bande de bruit de référence)
- Expérience régime : `python regime_experiment.py` (~35 min ; features via
  `DataConfig(regime_features=True)` + `EnvConfig(features=...)`, obs 52→72)
- **Convention expériences** : un modèle candidat s'entraîne dans SON dossier
  (`models/ppo_multi_regime/`, `models/seeds/…`) et n'est promu headline
  (`models/ppo_multi/`) que s'il gagne HORS bande de bruit inter-seeds.
- venv du projet : `.venv/` (Python 3.12) — pas d'autre environnement.

## Chantier diffusion (Phase 1 — générateur de scénarios)

Décision d'origine (juillet 2026, après l'Acte 3) : avant tout MARL, construire un
**DDPM sur séries de RENDEMENTS** (jamais de prix bruts) comme **livrable autonome**.
Positionnement : volet **IA générative** du portfolio de Martin (modélisation
générative appliquée aux marchés) — présenté comme tel dans le rapport et le CV.
Motivation : le walk-forward montre que l'agent rate les régimes rares (rebond 2020)
faute d'en avoir vus à l'entraînement → génération de scénarios = augmentation de
données motivée par le diagnostic, pas par la mode.

**STATUT (juillet 2026) : Phase 1 EXÉCUTÉE — NO-GO documenté ×2, Phase 2 verrouillée.**
Le générateur (package `diffusion/`, scripts `train_diffusion.py` /
`validate_diffusion.py`) et son protocole pré-enregistré existent et sont testés.
Récit complet : rapport Acte 4 (`sec:diffusion`).

- **Protocole FIGÉ** : bandes/ancres dans `reports/diffusion_validation.json` — ne
  PAS relancer `--calibration-only` sans raison (contrat de pré-enregistrement) ;
  verdict v1 archivé dans `diffusion_validation_v1.json`. Juger un nouveau candidat :
  `python validate_diffusion.py --model-dir models/<exp>/`.
- **Acquis** : É3/É4/É6 passent (volatility clustering capturé : ACF|r| lag1 0.143
  dans la bande réelle ; pas de mémorisation). **Échecs** : É2 (σ générée ×0.56,
  queue de vol ×0.43 — sous-engagement d'échelle sur le mélange multi-tickers) et
  É5 (disc 0.868 vs seuil 0.682) — identiques en ε-pred (v1) et v-pred (v2) ;
  sampler disculpé par test-oracle analytique (test permanent de la suite).
- **Levier v3 identifié, NON lancé (budget 2 itérations épuisé)** : retirer le
  mélange d'échelles (normalisation PAR TICKER, échelle réinjectée au tirage) ou
  conditionnement par la vol de fenêtre. À lancer uniquement comme nouvelle question
  de rapport, avec nouvelle prédiction pré-enregistrée.
- **Aucun branchement RL tant que les samples ne passent pas É1-É6** (piège n°1
  inchangé) ; Phase 2 = curriculum réel/synthétique jugé contre bande inter-seeds
  + walk-forward.
- Conventions instanciées : `PREDICTION.md` AVANT chaque run dans le dossier
  d'expérience (`models/diffusion_v1/`, `models/diffusion_v2/`) ; coût entraînement
  MPS : ~75 min (25k steps) à ~230 min (50k).
- Références DANS le repo : `2303.04137v5.pdf` (Diffusion Policy),
  `2510.12253v1.pdf` (survey Diffusion×RL).

## Pièges connus

- **Toujours normaliser les observations** avant `model.predict` : passer par
  `evaluate.load_model_and_norm()` qui charge le `vec_normalize.pkl` adjacent au modèle.
  Prédire sur obs brutes = résultats faux (bug historique corrigé 2 fois).
- `EVAL_ENV_CFG` (evaluate.py) doit rester **alignée sur la config d'entraînement**
  (train.py `__main__`) : frais 0.001, window 10, drawdown 25 %.
- `FEATURES` est dupliqué dans `data_loader.py` et `environment.py` — un test vérifie
  leur synchronisation ; modifier les deux ensemble.
- En multi-ticker, `segment_id` doit exister sur **les 3 splits** (train/val/test), sinon
  les épisodes traversent les frontières entre tickers (faux krachs → sélection de modèle
  corrompue).
- `overfitting.py` est un shim déprécié qui délègue à `evaluate.py` — ne pas y remettre
  de logique.
- Anciens modèles (2018-2023) archivés dans `models/archive_2018/` — ne pas écraser.
- Dates : source de vérité unique = `DataConfig` (data_loader.py, 2010→2023) ; ne pas
  re-hardcoder de dates dans les scripts.

## Style

- Pas de `Co-Authored-By: Claude` dans les commits ; commit uniquement à la demande.

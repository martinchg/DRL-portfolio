# 🧭 Projets futurs — après DRL Portfolio

> Fichier distinct de `roadmap.md` (qui reste le plan de CE projet : MARL puis diffusion).
> Ici : des projets **connexes mais indépendants**, choisis pour un profil candidat
> Quant, chacun relié à tes cours (économétrie, gestion de portefeuille, calcul
> stochastique) et à ce que tu as déjà construit ici.
>
> Pour chaque projet : le verdict modèle à utiliser avec Claude Code, selon ta
> convention (Sonnet élevé / Sonnet max / Opus).

---

## Comment choisir le prochain (la méthode, pas la liste)

Trois questions, dans cet ordre :
1. **Qu'est-ce que ça m'apprend que je ne sais pas déjà ?** (un 2ᵉ projet RL equity
   t'apprendrait peu ; un projet dérivés/vol ouvre un domaine entier)
2. **Est-ce que je peux le valider rigoureusement ?** (un projet sans protocole
   d'évaluation propre ne vaut rien en entretien — leçon n°1 de DRL Portfolio)
3. **Est-ce que ça raconte une histoire différente sur le CV ?** (diversifier :
   un projet *pricing/hedging*, un projet *économétrie*, un projet *exécution*)

Ordre recommandé : **1 → 3 → 5** (hedging = nouveau domaine + ton papier SABR ;
puis stat-arb = l'économétrie en vitrine ; puis allocation = extension naturelle d'ici).

---

## 1. Deep Hedging — couvrir une option par RL 🥇

**Pitch.** Entraîner un agent à couvrir une option vendue (call européen) en présence
de frais de transaction, et le comparer au delta-hedge de Black-Scholes. Papier
fondateur : *Deep Hedging* (Buehler, Gonon, Teichmann, Wood, 2019).

- **Ce que tu apprends** : pricing par réplication, grecques en pratique, pourquoi le
  delta-hedge théorique se dégrade avec les frais, objectifs en risque (CVaR comme
  fonction de perte d'entraînement, pas juste comme métrique).
- **Lien avec tes cours** : calcul stochastique (BS, EDS), gestion des risques
  (mesures cohérentes), et ton papier `pdf_SABR.pdf` — v2 du projet : couvrir sous
  vol stochastique SABR au lieu de BS, là où le delta-hedge naïf casse vraiment.
- **Lien avec DRL Portfolio** : tu réutilises TOUT (env Gymnasium custom, PPO, protocole
  d'éval déterministe, suite de tests) — seul le marché simulé change (trajectoires
  Monte-Carlo au lieu de données historiques → plus de données infinies, moins de
  problèmes d'overfitting : bon contraste à raconter).
- **Effort** : 3-4 semaines. **CV** : très fort — c'est le sujet RL que les desks
  dérivés connaissent.
- **Verdict modèle** : **Opus** pour la conception (formulation reward/CVaR, design de
  l'env), puis **Sonnet élevé** pour l'implémentation et les tests.

## 2. Calibration SABR et dynamique du smile 📐

**Pitch.** Implémenter la formule de Hagan (ton `pdf_SABR.pdf`), calibrer (α, β, ρ, ν)
sur des surfaces de vol réelles (options SPY/SPX ou taux), visualiser la dynamique du
smile, et vérifier empiriquement la critique du papier (vol locale vs SABR : le smile
bouge dans le mauvais sens).

- **Ce que tu apprends** : vol implicite, smile/skew, calibration = moindres carrés
  non linéaires (économétrie non linéaire !), risques vanna/volga.
- **Lien avec tes cours** : le pont parfait calcul stochastique ↔ économétrie
  (estimation de paramètres d'une EDS sur données de marché).
- **Effort** : 2-3 semaines, pas de ML — que du numérique propre.
- **Verdict modèle** : **Sonnet max** (calibration numérique délicate mais bien délimitée).

## 3. Stat-arb : pairs trading par cointégration 📈📉

**Pitch.** Sélectionner des paires cointégrées (Engle-Granger, Johansen), modéliser le
spread comme un processus de retour à la moyenne (Ornstein-Uhlenbeck), trader les
z-scores, et backtester avec la MÊME rigueur qu'ici (walk-forward, frais, tests).

- **Ce que tu apprends** : la différence stationnarité/cointégration en pratique, le
  danger des régressions fallacieuses, l'estimation d'un OU (vitesse de retour κ,
  demi-vie du spread), et pourquoi les paires « meurent » (rupture structurelle
  — test de Chow, ta matière).
- **Lien avec tes cours** : c'est LE projet d'économétrie appliquée — tout le
  vocabulaire du cours devient un signal de trading.
- **Lien avec DRL Portfolio** : réutilise `walk_forward.py`, les métriques
  d'`evaluate.py` et la discipline anti look-ahead telles quelles.
- **Effort** : 2-3 semaines. **Verdict modèle** : **Sonnet élevé**.

## 4. Prévision de volatilité : GARCH vs machine learning 🌪️

**Pitch.** Prévoir la variance réalisée à 1 jour / 1 semaine : GARCH(1,1), EGARCH,
GJR (asymétrie) contre LSTM/HAR-RV, évalués proprement (QLIKE, test de
Diebold-Mariano) en walk-forward.

- **Ce que tu apprends** : pourquoi GARCH reste un benchmark redoutable, l'évaluation
  de prévisions de variance (les MSE naïfs mentent), les fonctions de perte robustes.
- **Lien avec DRL Portfolio** : ta feature `volatility` (proxy naïf) serait remplacée
  par une vraie prévision — et tu peux mesurer si l'agent PPO en profite (ablation).
- **Effort** : 2 semaines. **Verdict modèle** : **Sonnet élevé**.

## 5. Allocation de portefeuille sous contrainte CVaR ⚖️

**Pitch.** Passer du signal mono-actif à l'ALLOCATION : poids continus sur N actifs,
comparaison à trois étages — Markowitz (échantillon), Black-Litterman (vues),
agent RL à actions continues — sous contrainte de CVaR et de turnover.

- **Ce que tu apprends** : l'instabilité de Markowitz (erreur d'estimation de µ),
  pourquoi les praticiens régularisent (shrinkage de Ledoit-Wolf), l'optimisation
  CVaR de Rockafellar-Uryasev (un programme linéaire élégant).
- **Lien avec tes cours** : gestion de portefeuille frontale, du cours au code.
- **Lien avec DRL Portfolio** : c'est l'évolution naturelle de l'env actuel
  (action `Box(-1,1)^N` au lieu de 4 actions discrètes) — résout au passage la
  limite « position tout-ou-rien » du rapport.
- **Effort** : 3 semaines. **Verdict modèle** : **Sonnet max**.

## 6. Market making : Avellaneda-Stoikov et carnet d'ordres 🏦

**Pitch.** Simuler un carnet d'ordres simple, implémenter le market maker
d'Avellaneda-Stoikov (solution analytique !), puis le comparer à un agent RL qui
apprend les quotes optimales sous risque d'inventaire.

- **Ce que tu apprends** : microstructure (spread, adverse selection, inventaire),
  contrôle stochastique appliqué (HJB), et la comparaison honnête
  « solution fermée vs RL » — un excellent sujet d'entretien.
- **Effort** : 4+ semaines, le plus difficile de la liste.
- **Verdict modèle** : **Opus** (conception env + lien HJB), Sonnet élevé ensuite.

---

## Et tes références actuelles ?

- **Diffusion Policy** (2303.04137) et **Diffusion Models for RL** (2510.12253) →
  nourrissent la **phase 2 de la roadmap de CE projet** (génération de scénarios
  synthétiques pour l'entraînement), pas un projet séparé.
- **State Representation Learning for Deep RL** (2506.17518) → pertinent ici aussi :
  une itération future = améliorer l'observation (features apprises, auto-encodeur
  sur la fenêtre de marché) plutôt que les features à la main. À faire APRÈS le
  walk-forward, avec ablation propre. **Verdict modèle : Sonnet max.**
- **Managing Smile Risk** (Hagan, `pdf_SABR.pdf`) → projets 1 et 2 ci-dessus.

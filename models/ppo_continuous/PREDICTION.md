# Prédiction pré-enregistrée — Acte 5, bras A et B (avant les runs réels)

Date : 2026-07-05, AVANT `python continuous_experiment.py` (runs réels 800k/400k).
Convention du repo : prédiction datée avant chaque run, confrontée au mesuré.

## Config jugée

- Bras A : `EnvConfig(continuous=True)` — poids w ∈ [-1,1], comptabilité
  cash/shares conservée, bande de non-rebalancement 0.5 %, récompense
  INCHANGÉE (α − frais − 0.05·dd). ent_coef 0.05 → 0.01 (gaussienne).
- Bras B : A + `risk_aversion=0.1` → pénalité 0.1·σ̂·|w|·100 (≈ 0.12·|w| au
  σ̂ médian — ~10 % du reward typique). UNE valeur de λ, pas de tuning.
- Juges : cross-ticker test (bande inter-seeds ±6.5 pts) + walk-forward
  5 folds (mêmes folds, mêmes seeds, ent_coef propagé).

## Critères figés

C1 cross ≥ +10.7 % · C2 fold 2020 > −50 % · C3 fold 2022 > +20 % ·
C4 médiane WF ≥ −2 %. Budget : ces deux bras, AUCUN bras C improvisé.

## Ce que je prédis

1. **Bras A : C1 ✅, C3 ✅, C2 ❌ (amélioré mais insuffisant).** Le sizing
   continu module l'exposition (la gaussienne apprend des poids moyens
   < 1), donc le drawdown 2020 mord moins vite : fold 2020 attendu
   **−80 % ± 30** (vs −146 %) — mieux, mais rien dans la récompense ne paie
   le dérisquage RAPIDE : je ne crois pas au passage sous −50 % sans
   incitation explicite.
2. **Bras B : LE test de l'hypothèse de l'Acte 3** (« l'incitation prime sur
   l'information »). La pénalité σ̂·|w| rend le portage coûteux dès que la
   vol monte → dérisquage avant le kill-switch. C2 attendu nettement
   meilleur que A ; passage sous −50 % plausible. **Risque principal : C3**
   — trop de prudence rabote l'alpha défensif 2022 (le short profitable de
   2022 est AUSSI du |w| en période nerveuse, la pénalité ne distingue pas
   le sens). C'est la vraie tension du design.
3. **Turnover** : A tradera plus que le discret (jitter résiduel malgré la
   bande) ; B moins que A (|w| pénalisé). n_trades attendus : A > 300,
   B < A.
4. **Verdict global prédit : A NO-GO (C2), B GO à ~50 %** — suspendu à
   l'arbitrage C2/C3. Si B passe C2 mais casse C3 : résultat documenté, le
   levier suivant serait une pénalité ASYMÉTRIQUE (σ̂ conditionnée au signe
   du marché), hors budget de cet acte.

## Ce qui me ferait dire que je me suis trompé utilement

- A passe C2 seul → le sizing suffisait, l'histoire « incitation » était
  superflue — plus simple gagne.
- B pire que A sur C2 → la pénalité pousse à ne JAMAIS s'exposer (flat
  permanent), signe que λ=0.1 est trop fort — à documenter, pas à retuner.

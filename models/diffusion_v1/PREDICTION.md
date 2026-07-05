# Prédiction pré-enregistrée — DDPM v1 (avant le run d'entraînement)

Date : 2026-07-04, AVANT `python train_diffusion.py` (convention du repo :
prédiction écrite avant chaque run, confrontée ensuite au mesuré).

Config jugée : U-Net 1D (32-64-128, ~1.9 M params), T=1000 cosine, ε-prediction,
EMA 0.999, 25 000 steps, batch 128, fenêtres 256 j normalisées globalement,
protocole v2 (bandes par blocs, É4 asymétrique, juge 3-seeds clippé).

## Ce que je prédis

1. **É2 (queues) : PASSE.** La kurtosis poolée sera dans la bande réelle —
   les DDPM reproduisent bien les marginales quand la normalisation est
   globale (le mélange de régimes de vol fait le gros des queues).
2. **É3 (ACF parasite) : PASSE.** Aucun mécanisme ne crée d'autocorrélation
   linéaire : le U-Net apprend des textures locales, pas un drift signé.
3. **É4 (clustering) : LE POINT DE FRICTION.** ACF(|r|) lag 1 positive et
   probablement dans la bande, mais décroissance trop RAPIDE aux lags > 10 :
   la persistance longue est la structure la plus dure à capturer pour un
   U-Net court (champ réceptif effectif ~60 j, pas d'attention). Si É4
   échoue, ce sera par somme trop basse, PAS par absence de lag 1.
4. **É5 (discriminatif) : entre bootstrap et GARCH.** Le juge (0.97 sur le
   bootstrap, ~0.64 sur GARCH en v1) trouvera quelque chose : je prédis
   acc DDPM ∈ [0.60, 0.75], c.-à-d. échec possible de la marge +0.05 vs
   GARCH à la v1.
5. **É6 (mémorisation) : PASSE.** 10 000 fenêtres d'entraînement pour ~1.9 M
   params et un objectif bruité : le régime est celui de la généralisation,
   pas de la copie (médiane NN nettement > p10 réel = 6.91).

## Verdict global prédit

**NO-GO en v1** (É4 et/ou É5), avec marginales et anti-mémorisation propres.
Leviers v2 déjà identifiés si É4 échoue comme prédit : +1 niveau de U-Net ou
attention au bottleneck (champ réceptif), T réduit à 500 (moins de diffusion
du signal basse fréquence), fenêtres 512 j.

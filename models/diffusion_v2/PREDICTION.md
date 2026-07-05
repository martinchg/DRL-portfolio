# Prédiction pré-enregistrée — DDPM v2 (avant le run d'entraînement)

Date : 2026-07-04, AVANT `python train_diffusion.py --pred-type v --steps 50000
--out-dir models/diffusion_v2`. Itération 2/2 du budget pré-enregistré.

## Ce que la v1 a mesuré (NO-GO, 3/5)

É3 ✅, É4 ✅ (clustering capturé — ma prédiction v1 d'un échec É4 était FAUSSE),
É6 ✅ ; É2 ❌ (kurt 35.8 > 20.1) et É5 ❌ (disc 0.868 > 0.682). Cause racine
diagnostiquée : variance échantillonnée rétrécie de 45 % (z-std 0.52) — biais
minuscule de l'ε-prediction composé sur T=1000 pas, données quasi blanches.
Expérience contrôlée sur GARCH connu : ε-pred → z-std 1.293 ; v-pred → 0.945.

## Deltas v2 (et rien d'autre)

1. pred_type = 'v' (Salimans & Ho 2022) — le remède structurel mesuré ;
2. steps 25k → 50k (la loss v1 descendait encore ; les queues s'apprennent
   en dernier) ;
3. LR cosine 2e-4 → 1e-5 (calibration fine de fin d'entraînement).

## Ce que je prédis

1. **É2 : PASSE.** z-std ∈ [0.9, 1.05] → std poolée ≈ 0.017-0.019 ; la
   kurtosis redescend dans la bande [3.7, 20.1] (l'excès v1 venait de spikes
   relatifs à une échelle fausse, pas d'un vrai régime de queues).
2. **É3 : PASSE** (aucun mécanisme nouveau d'autocorrélation linéaire).
3. **É4 : PASSE** (déjà acquis en v1 ; v-pred ne détruit pas la structure
   de clustering, elle améliore le conditionnement).
4. **É5 : LE POINT INCERTAIN.** L'échelle corrigée enlève l'indice trivial du
   juge ; je prédis acc ∈ [0.62, 0.72] — la marge vs GARCH (≤ 0.682) se joue
   à quelques points. C'est le critère qui décidera GO/NO-GO.
5. **É6 : PASSE** (régime de généralisation inchangé).

## Verdict global prédit

**GO probable mais non certain (~60 %)**, suspendu à É5. Si NO-GO : c'est un
échec DOCUMENTÉ après 2 itérations sérieuses — on écrit le résultat négatif
au rapport (protocole robuste, générateur pas encore au niveau de sa baseline
classique au juge discriminatif) et on n'itère pas en v3 sauvage.

# overfitting.py — DÉPRÉCIÉ
#
# Ce script était l'ancienne version du diagnostic d'overfitting.
# Il prédisait sur des observations NON normalisées (sans vec_normalize.pkl)
# et supposait 3 actions au lieu de 4 → métriques trompeuses.
#
# Tout est maintenant dans evaluate.py :
#   - check_overfitting_both()  → diagnostic Train/Val/Test (obs normalisées)
#   - compare_models()          → Single vs Multi vs B&H
#   - generalization_report()   → généralisation cross-ticker
#
# Ce fichier ne fait plus que déléguer, pour ne pas casser `python overfitting.py`.
from evaluate import check_overfitting_both, compare_models  # noqa: F401

if __name__ == "__main__":
    print("⚠️  overfitting.py est déprécié → utilisation d'evaluate.py\n")
    check_overfitting_both()

# Pistes d'amélioration — Équation de Nagumo

## C'est quoi
Projet de modélisation/simulation numérique de l'équation de Nagumo (EDP
instationnaire) : notebook de groupe + article + scripts de discrétisation.

## État actuel
- `notebook-groupe3/` : `simulation_final.ipynb`, beamer et PDF du modèle.
- `sources-article/` : `nagumo_modeling.tex`, `nagumo.bib`, PDFs de référence, `nagumo_simulation.m`.
- `scripts/` : `nagumo.py`, `nagumo2.py`, `mef.py`, `hagumop.py`, euler implicite -ADI, etc.

## Pistes (courtes)
- Unifier les scripts en un seul module clair (`nagumo.py`) + un `main` par schéma.
- Ajouter un `requirements.txt` (numpy, scipy, matplotlib).
- Corriger les noms (`hagumop.py`, `nagumo2.py`) et documenter chaque schéma (Euler implicite, ADI, éléments finis).
- Ajouter un `README.md` : équation, schémas numériques, comment lancer, figures obtenues.
- Vérifier la convergence (pas de temps/espace) et l'ajouter au README.

## Mini-prompt
> Regroupe les scripts en un module unique documenté, ajoute `requirements.txt`
> et un README (équation, schémas, exécution, convergence).

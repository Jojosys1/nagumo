# Équation de Nagumo — problème instationnaire

Modélisation et **simulation numérique** de l'équation de Nagumo (réaction-diffusion)
en 2D :

$$u_t = D\,\Delta u + u(1-u)(u-a), \qquad \text{Neumann homogènes sur le bord.}$$

Travail de groupe (ENSGMM) : notebook de simulation, article de modélisation et
scripts comparant plusieurs schémas numériques.

## Contenu

| Dossier | Contenu |
|---|---|
| `notebook-groupe3/` | `simulation_final.ipynb` (notebook final), beamer, PDF du modèle, sources LaTeX (`img/`) |
| `sources-article/` | Article `nagumo_modeling.tex` + `nagumo.bib`, code MATLAB `nagumo_simulation.m`, notebook `simulation.ipynb` |
| `scripts/` | Scripts Python des schémas numériques |

## Schémas numériques comparés

- `nagumo.py` : comparaison RK4, Euler implicite (ADI), AB2, AM2 + convergence.
- `nagumo2.py` : version corrigée (corrections documentées dans l'entête).
- `euler_implicite_ADI.py` : Euler implicite semi-implicite, méthode ADI de Peaceman-Rachford (avec `numba`).
- `mef.py` : éléments finis P1 en 1D (from scratch, numpy uniquement).
- `hagumop.py` : variante de simulation.
- `question_discretisation.txt` : note sur la discrétisation.

## Prérequis

```bash
pip install -r requirements.txt
```

(`numpy`, `scipy`, `matplotlib`, `numba`, `jupyter`.)

## Utilisation

```bash
python scripts/nagumo.py
python scripts/euler_implicite_ADI.py
jupyter notebook notebook-groupe3/simulation_final.ipynb
pdflatex sources-article/nagumo_modeling.tex   # article
```

## Remarques

- Certains scripts contiennent des accents mal encodés dans les commentaires
  (mojibake) : à nettoyer.
- `hagumop.py` mériterait un nom plus explicite.

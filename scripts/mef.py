"""
============================================================
SECTION 1 — Implémentation MEF P1 en 1D (from scratch)
Équation : -u''(x) = 1,  u(0) = u(1) = 0
Solution exacte : u(x) = x(1-x)/2
============================================================
Aucune bibliothèque MEF — uniquement numpy.
"""

import numpy as np

# ─────────────────────────────────────────────
# 1.1  Maillage uniforme
# ─────────────────────────────────────────────
def make_mesh_1d(N):
    """
    Découpe [0,1] en N éléments P1.
    Retourne :
      nodes   : coordonnées des N+1 nœuds
      elems   : connectivité (N, 2) — indices des nœuds de chaque élément
    """
    nodes = np.linspace(0.0, 1.0, N + 1)   # x_0, x_1, …, x_N
    elems = np.column_stack([               # élément e = [nœud gauche, nœud droit]
        np.arange(N),
        np.arange(1, N + 1)
    ])
    return nodes, elems


# ─────────────────────────────────────────────
# 1.2  Matrice de rigidité élémentaire
# ─────────────────────────────────────────────
def elem_stiffness(h):
    """
    Kᵉ = ∫₀ʰ φᵢ'·φⱼ' dx   (base locale {φ₁, φ₂} linéaires sur [0,h])
    φ₁(x) = 1 - x/h,  φ₂(x) = x/h
    φ₁' = -1/h,        φ₂' = 1/h
    → Kᵉ = (1/h) * [[1, -1], [-1, 1]]
    """
    return (1.0 / h) * np.array([[ 1., -1.],
                                  [-1.,  1.]])


# ─────────────────────────────────────────────
# 1.3  Vecteur de charge élémentaire
# ─────────────────────────────────────────────
def elem_load(h, f_values):
    """
    Fᵉ = ∫₀ʰ f(x)·φᵢ(x) dx
    Quadrature de Gauss à 2 points (exacte pour f·φ quadratique) :
      ξ₁ = (1 - 1/√3)/2,  ξ₂ = (1 + 1/√3)/2   (coord. locales ∈ [0,1])
      w₁ = w₂ = 1/2
    f_values = [f(x_a), f(x_b)]  (valeurs aux nœuds de l'élément)
    """
    # Points et poids de Gauss sur [0,1]
    xi  = np.array([(1 - 1/np.sqrt(3)) / 2,
                    (1 + 1/np.sqrt(3)) / 2])
    w   = np.array([0.5, 0.5])

    # Fonctions de forme locales évaluées aux points de Gauss
    phi = np.array([1 - xi, xi])   # phi[0] = φ₁(ξ), phi[1] = φ₂(ξ)

    # Interpoler f aux points de Gauss
    f_gp = f_values[0] * phi[0] + f_values[1] * phi[1]

    # Intégrale ≈ Σ w_g * f(ξ_g) * φᵢ(ξ_g) * h
    Fe = h * np.array([
        np.sum(w * f_gp * phi[0]),
        np.sum(w * f_gp * phi[1])
    ])
    return Fe


# ─────────────────────────────────────────────
# 1.4  Assemblage global
# ─────────────────────────────────────────────
def assemble_1d(nodes, elems, f_func):
    """
    Assemble la matrice globale K (n_nodes×n_nodes)
    et le vecteur F (n_nodes,) par boucle sur les éléments.
    """
    n  = len(nodes)
    K  = np.zeros((n, n))
    F  = np.zeros(n)

    for e, (i, j) in enumerate(elems):
        h     = nodes[j] - nodes[i]        # longueur de l'élément
        Ke    = elem_stiffness(h)
        f_val = np.array([f_func(nodes[i]),
                          f_func(nodes[j])])
        Fe    = elem_load(h, f_val)

        # Ajout (scatter) dans les matrices globales
        idx = [i, j]
        for r, gi in enumerate(idx):
            F[gi] += Fe[r]
            for s, gj in enumerate(idx):
                K[gi, gj] += Ke[r, s]

    return K, F


# ─────────────────────────────────────────────
# 1.5  Conditions aux limites de Dirichlet (élimination directe)
# ─────────────────────────────────────────────
def apply_dirichlet(K, F, fixed_dofs, fixed_vals):
    """
    Applique u(dof) = val en modifiant K et F :
      - ligne/colonne du dof → identité
      - F des autres dofs corrigé par la valeur imposée
    """
    n    = K.shape[0]
    Kmod = K.copy()
    Fmod = F.copy()

    for dof, val in zip(fixed_dofs, fixed_vals):
        # Correction du second membre pour les lignes libres
        Fmod -= Kmod[:, dof] * val
        Fmod[dof] = val

        # Mise à zéro de la ligne et colonne + identité
        Kmod[dof, :] = 0.0
        Kmod[:, dof] = 0.0
        Kmod[dof, dof] = 1.0

    return Kmod, Fmod


# ─────────────────────────────────────────────
# 1.6  Résolution et affichage
# ─────────────────────────────────────────────
def solve_mef_1d(N, f_func, u_exact):
    """Pipeline complet : maillage → assemblage → CL → résolution."""
    nodes, elems = make_mesh_1d(N)

    K, F = assemble_1d(nodes, elems, f_func)
    Kmod, Fmod = apply_dirichlet(K, F,
                                  fixed_dofs=[0, N],
                                  fixed_vals=[0.0, 0.0])

    # Résolution directe (numpy.linalg.solve = LU interne)
    U = np.linalg.solve(Kmod, Fmod)

    # Erreurs
    u_ex = u_exact(nodes)
    err_L2 = compute_error_L2(nodes, elems, U, u_exact)
    err_H1 = compute_error_H1(nodes, elems, U, u_exact)

    return nodes, U, u_ex, err_L2, err_H1


# ─────────────────────────────────────────────
# 1.7  Calcul des erreurs L² et H¹ (norme continue)
# ─────────────────────────────────────────────
def compute_error_L2(nodes, elems, U, u_exact):
    """
    ‖u - uₕ‖²_L² = Σₑ ∫ₑ (u - uₕ)² dx
    Gauss à 3 points (exacte pour polynômes deg ≤ 5).
    """
    xi_g = np.array([0.5 - np.sqrt(15)/10,
                     0.5,
                     0.5 + np.sqrt(15)/10])
    w_g  = np.array([5/18, 4/9, 5/18])

    err2 = 0.0
    for i, j in elems:
        h   = nodes[j] - nodes[i]
        x_g = nodes[i] + h * xi_g          # points physiques

        # uₕ aux points de Gauss (interpolation P1)
        uh_g = U[i] * (1 - xi_g) + U[j] * xi_g

        diff = u_exact(x_g) - uh_g
        err2 += h * np.sum(w_g * diff**2)

    return np.sqrt(err2)


def compute_error_H1(nodes, elems, U, du_exact):
    """
    |u - uₕ|²_H¹ = Σₑ ∫ₑ (u' - uₕ')² dx
    uₕ' est constant par élément (P1).
    Solution exacte : u'(x) = (1-2x)/2  pour -u''=1.
    On recalcule du_exact à partir de la fonction u_exact passée.
    """
    xi_g = np.array([0.5 - np.sqrt(15)/10,
                     0.5,
                     0.5 + np.sqrt(15)/10])
    w_g  = np.array([5/18, 4/9, 5/18])

    err2 = 0.0
    for i, j in elems:
        h     = nodes[j] - nodes[i]
        uh_dx = (U[j] - U[i]) / h          # dérivée de uₕ (constante sur l'élément)

        x_g   = nodes[i] + h * xi_g
        # dérivée exacte par différence finie (précision numérique suffisante)
        eps   = 1e-8
        du_g  = (du_exact(x_g + eps) - du_exact(x_g - eps)) / (2 * eps)

        diff  = du_g - uh_dx
        err2 += h * np.sum(w_g * diff**2)

    return np.sqrt(err2)


# ─────────────────────────────────────────────
# Problème : -u'' = 1,  u(0)=u(1)=0
# Solution exacte : u(x) = x(1-x)/2
# ─────────────────────────────────────────────
f_func  = lambda x: np.ones_like(np.atleast_1d(x), dtype=float)
u_exact = lambda x: x * (1 - x) / 2.0

if __name__ == "__main__":
    N = 8
    nodes, Uh, u_ex, eL2, eH1 = solve_mef_1d(N, f_func, u_exact)

    print(f"\n{'─'*50}")
    print(f"  MEF P1 1D  |  N={N} éléments  |  h={1/N:.4f}")
    print(f"{'─'*50}")
    print(f"  Nœuds       : {nodes.round(4)}")
    print(f"  Uₕ          : {Uh.round(6)}")
    print(f"  u_exacte    : {u_ex.round(6)}")
    print(f"  ‖u−uₕ‖_L²  : {eL2:.2e}")
    print(f"  |u−uₕ|_H¹  : {eH1:.2e}")
    print(f"{'─'*50}")
    
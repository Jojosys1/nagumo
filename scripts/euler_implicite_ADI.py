"""
Résolution de l'équation de Nagumo 2D par Euler Implicite (semi-implicite)
avec méthode ADI de Peaceman-Rachford
"""

import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from numba import njit, prange
import time


# =============================================================================
# PARAMÈTRES PHYSIQUES ET NUMÉRIQUES
# =============================================================================

class NagumoParams:
    """Paramètres de l'équation de Nagumo : u_t = D*Δu + f(u)"""
    D = 1.0          # Coefficient de diffusion
    a = 0.25         # Paramètre de la non-linéarité (0 < a < 1)
    
    # Domaine
    Lx, Ly = 20.0, 20.0
    T_final = 500.0   # Temps final
    
    # Discrétisation
    Nx, Ny = 200, 200
    dt = 0.1         # Pas de temps (stable car implicite)
    
    @property
    def dx(self):
        return self.Lx / self.Nx
    
    @property
    def dy(self):
        return self.Ly / self.Ny
    
    @property
    def mu(self):
        """Nombre de Fourier : μ = D*dt/h²"""
        return self.D * self.dt / (self.dx ** 2)


def f_nagumo(u, a=0.25):
    """Terme réaction : f(u) = u(1-u)(u-a)"""
    return u * (1.0 - u) * (u - a)


def f_prime_nagumo(u, a=0.25):
    """Dérivée : f'(u) = -3u² + 2(1+a)u - a"""
    return -3*u**2 + 2*(1+a)*u - a


# =============================================================================
# MÉTHODE 1 : EULER IMPLICITE DIRECT (matrice creuse)
# =============================================================================

def build_laplacian_2d(Nx, Ny, dx, dy):
    """
    Construction de la matrice du Laplacien discret 2D avec conditions de Neumann
    (ou Dirichlet homogène) par produit de Kronecker.
    """
    # Laplacien 1D en x
    main_x = -2.0 * np.ones(Nx)
    off_x = np.ones(Nx - 1)
    Ax = diags([off_x, main_x, off_x], [-1, 0, 1], shape=(Nx, Nx)) / (dx**2)
    
    # Laplacien 1D en y
    main_y = -2.0 * np.ones(Ny)
    off_y = np.ones(Ny - 1)
    Ay = diags([off_y, main_y, off_y], [-1, 0, 1], shape=(Ny, Ny)) / (dy**2)
    
    # Laplacien 2D : A_2D = I_y ⊗ Ax + Ay ⊗ I_x
    Ix = eye(Nx)
    Iy = eye(Ny)
    A_2D = kron(Iy, Ax) + kron(Ay, Ix)
    
    return A_2D


def euler_implicite_direct(U0, params):
    """
    Euler implicite direct : (I - dt*A)U^{n+1} = U^n + dt*f(U^n)
    Résolution par spsolve (matrice creuse).
    """
    Nx, Ny = params.Nx, params.Ny
    dx, dy, dt = params.dx, params.dy, params.dt
    D, a = params.D, params.a
    nt = int(params.T_final / dt)
    
    # Construction de la matrice système
    A_2D = build_laplacian_2d(Nx, Ny, dx, dy)
    M = eye(Nx * Ny) - dt * D * A_2D
    
    U = U0.flatten().copy()
    history = [U.reshape((Ny, Nx)).copy()]
    
    t0 = time.time()
    for n in range(nt):
        # Terme réaction explicite
        F = f_nagumo(U.reshape((Ny, Nx)), a).flatten()
        rhs = U + dt * F
        
        # Résolution du système linéaire creux
        U = spsolve(M, rhs)
        
        if (n + 1) % max(1, nt // 20) == 0:
            history.append(U.reshape((Ny, Nx)).copy())
    
    elapsed = time.time() - t0
    print(f"Euler implicite direct : {elapsed:.2f}s ({nt} pas)")
    
    return U.reshape((Ny, Nx)), history


# =============================================================================
# MÉTHODE 2 : ADI (Alternating Direction Implicit) - OPTIMISÉE
# =============================================================================

@njit(parallel=True, cache=True)
def solve_tridiag_parallel(a, b, c, rhs):
    """
    Résolution de systèmes tridiagonaux en parallèle (une ligne/colonne à la fois).
    a, b, c : diagonales (b = principale), shape = (n_systems, n)
    rhs : second membre, shape = (n_systems, n)
    """
    n_systems, n = rhs.shape
    x = np.empty_like(rhs)
    
    for s in prange(n_systems):
        # Copie locale pour éviter les conflits
        aa = a[s].copy()
        bb = b[s].copy()
        cc = c[s].copy()
        rr = rhs[s].copy()
        
        # Élimination de Gauss (Thomas algorithm)
        for i in range(1, n):
            w = aa[i-1] / bb[i-1]
            bb[i] -= w * cc[i-1]
            rr[i] -= w * rr[i-1]
        
        # Substitution arrière
        x[s, n-1] = rr[n-1] / bb[n-1]
        for i in range(n-2, -1, -1):
            x[s, i] = (rr[i] - cc[i] * x[s, i+1]) / bb[i]
    
    return x


def euler_implicite_adi(U0, params):
    """
    Méthode ADI de Peaceman-Rachford :
    - Demi-pas 1 : implicite en x, explicite en y + réaction
    - Demi-pas 2 : implicite en y, explicite en x
    """
    Nx, Ny = params.Nx, params.Ny
    dx, dy, dt = params.dx, params.dy, params.dt
    D, a = params.D, params.a
    nt = int(params.T_final / dt)
    
    mu_x = D * dt / (2 * dx**2)
    mu_y = D * dt / (2 * dy**2)
    
    # Matrices tridiagonales pour les demi-pas
    # Demi-pas 1 : (I - mu_x * A_x)  pour chaque ligne j
    main_x = (1 + 2*mu_x) * np.ones((Ny, Nx))
    off_x = -mu_x * np.ones((Ny, Nx-1))
    sub_x = -mu_x * np.ones((Ny, Nx-1))
    
    # Demi-pas 2 : (I - mu_y * A_y)  pour chaque colonne i
    main_y = (1 + 2*mu_y) * np.ones((Nx, Ny))
    off_y = -mu_y * np.ones((Nx, Ny-1))
    sub_y = -mu_y * np.ones((Nx, Ny-1))
    
    U = U0.copy()
    history = [U.copy()]
    
    t0 = time.time()
    for n in range(nt):
        # --- Demi-pas 1 : implicite en x, explicite en y + réaction ---
        # Second membre : (I + mu_y * A_y) * U^n + dt * f(U^n)
        F = f_nagumo(U, a)
        
        # Explicite en y : U + mu_y * (U_{j+1} - 2U_j + U_{j-1})
        rhs_half = U.copy()
        rhs_half[1:-1, :] += mu_y * (U[2:, :] - 2*U[1:-1, :] + U[:-2, :])
        rhs_half[[0, -1], :] += mu_y * (U[[1, -2], :] - U[[0, -1], :])  # Neumann
        rhs_half += dt * F
        
        # Résolution tridiagonale en x pour chaque ligne j
        U_half = solve_tridiag_parallel(
            sub_x, main_x, off_x, rhs_half
        )
        
        # --- Demi-pas 2 : implicite en y, explicite en x ---
        # Second membre : (I + mu_x * A_x) * U^{n+1/2}
        rhs_new = U_half.copy()
        rhs_new[:, 1:-1] += mu_x * (U_half[:, 2:] - 2*U_half[:, 1:-1] + U_half[:, :-2])
        rhs_new[:, [0, -1]] += mu_x * (U_half[:, [1, -2]] - U_half[:, [0, -1]])  # Neumann
        
        # Résolution tridiagonale en y pour chaque colonne i (transposée)
        U_new = solve_tridiag_parallel(
            sub_y, main_y, off_y, rhs_new.T
        ).T
        
        U = U_new
        
        if (n + 1) % max(1, nt // 20) == 0:
            history.append(U.copy())
    
    elapsed = time.time() - t0
    print(f"ADI (numba parallèle) : {elapsed:.2f}s ({nt} pas)")
    
    return U, history


# =============================================================================
# MÉTHODE 3 : ADI AVEC SCIPY BANDED (alternative sans numba)
# =============================================================================

def euler_implicite_adi_scipy(U0, params):
    """
    Version ADI utilisant solve_banded de scipy (sans numba).
    """
    Nx, Ny = params.Nx, params.Ny
    dx, dy, dt = params.dx, params.dy, params.dt
    D, a = params.D, params.a
    nt = int(params.T_final / dt)
    
    mu_x = D * dt / (2 * dx**2)
    mu_y = D * dt / (2 * dy**2)
    
    U = U0.copy()
    history = [U.copy()]
    
    t0 = time.time()
    for n in range(nt):
        F = f_nagumo(U, a)
        
        # --- Demi-pas 1 ---
        rhs_half = U.copy()
        rhs_half[1:-1, :] += mu_y * (U[2:, :] - 2*U[1:-1, :] + U[:-2, :])
        rhs_half += dt * F
        
        # Résolution par ligne avec solve_banded
        ab_x = np.zeros((3, Nx))
        ab_x[0, 1:] = -mu_x          # sur-diagonale
        ab_x[1, :] = 1 + 2*mu_x      # diagonale
        ab_x[2, :-1] = -mu_x         # sous-diagonale
        
        U_half = np.zeros_like(U)
        for j in range(Ny):
            U_half[j, :] = solve_banded((1, 1), ab_x, rhs_half[j, :])
        
        # --- Demi-pas 2 ---
        rhs_new = U_half.copy()
        rhs_new[:, 1:-1] += mu_x * (U_half[:, 2:] - 2*U_half[:, 1:-1] + U_half[:, :-2])
        
        ab_y = np.zeros((3, Ny))
        ab_y[0, 1:] = -mu_y
        ab_y[1, :] = 1 + 2*mu_y
        ab_y[2, :-1] = -mu_y
        
        for i in range(Nx):
            U[:, i] = solve_banded((1, 1), ab_y, rhs_new[:, i])
        
        if (n + 1) % max(1, nt // 20) == 0:
            history.append(U.copy())
    
    elapsed = time.time() - t0
    print(f"ADI (scipy banded) : {elapsed:.2f}s ({nt} pas)")
    
    return U, history


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_evolution(history, params, title=""):
    """Animation statique des étapes clés"""
    n_plots = min(6, len(history))
    indices = np.linspace(0, len(history)-1, n_plots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    vmin, vmax = -0.1, 1.1
    for idx, ax in zip(indices, axes):
        im = ax.imshow(history[idx], extent=[0, params.Lx, 0, params.Ly],
                       origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f't = {idx * params.T_final / (len(history)-1):.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/agents/output/nagumo_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_comparison(U_direct, U_adi, params):
    """Comparaison des deux méthodes et erreur"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin, vmax = -0.1, 1.1
    
    im0 = axes[0].imshow(U_direct, extent=[0, params.Lx, 0, params.Ly],
                         origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Euler Implicite Direct')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(U_adi, extent=[0, params.Lx, 0, params.Ly],
                         origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('ADI Peaceman-Rachford')
    plt.colorbar(im1, ax=axes[1])
    
    err = np.abs(U_direct - U_adi)
    im2 = axes[2].imshow(err, extent=[0, params.Lx, 0, params.Ly],
                         origin='lower', cmap='hot')
    axes[2].set_title(f'Erreur |Direct - ADI|\nMax = {err.max():.2e}')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('/mnt/agents/output/nagumo_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    params = NagumoParams()
    
    # Condition initiale : pulse gaussien
    x = np.linspace(0, params.Lx, params.Nx)
    y = np.linspace(0, params.Ly, params.Ny)
    X, Y = np.meshgrid(x, y)
    
    # Pulse initial centré
    U0 = np.exp(-((X - params.Lx/2)**2 + (Y - params.Ly/2)**2) / 2)
    
    print(f"Grille : {params.Nx}x{params.Ny}, μ = {params.mu:.3f}")
    print(f"Pas de temps : {params.dt}, Pas totaux : {int(params.T_final/params.dt)}")
    print("-" * 50)
    
    # Test ADI (méthode rapide)
    U_adi, hist_adi = euler_implicite_adi(U0, params)
    
    # Test avec grille plus petite pour la méthode directe (sinon trop lent)
    params_small = NagumoParams()
    params_small.Nx, params_small.Ny = 80, 80
    params_small.dt = 0.05
    
    x_s = np.linspace(0, params_small.Lx, params_small.Nx)
    y_s = np.linspace(0, params_small.Ly, params_small.Ny)
    X_s, Y_s = np.meshgrid(x_s, y_s)
    U0_small = np.exp(-((X_s - params_small.Lx/2)**2 + (Y_s - params_small.Ly/2)**2) / 2)
    
    print("\n" + "-" * 50)
    U_direct, hist_direct = euler_implicite_direct(U0_small, params_small)
    U_adi_small, _ = euler_implicite_adi(U0_small, params_small)
    
    # Comparaison
    plot_comparison(U_direct, U_adi_small, params_small)
    
    # Évolution ADI
    plot_evolution(hist_adi, params, "Évolution Nagumo 2D - ADI")
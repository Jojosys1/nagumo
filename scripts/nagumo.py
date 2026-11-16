# -*- coding: utf-8 -*-
"""
Simulation de l'equation de Nagumo 2D par 4 schemas:
   u_t = D*(u_xx + u_yy) + u*(1-u)*(u-a)
   Neumann homogenes sur tout le bord.
   Domain [0,Lx]x[0,Ly], t in [0,T].
   Comparaison: RK4, Euler implicite (ADI), AB2, AM2.
   Mesure des temps d'execution et verification de la convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import splu
import time
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# 1. Parametres physiques
# ------------------------------------------------------------
Lx, Ly = 60.0, 60.0
D = 1.0
a = 0.25
T = 20.0

# ------------------------------------------------------------
# 2. Discretisation spatiale
# ------------------------------------------------------------
Nx = 300
Ny = 300
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y)

# ------------------------------------------------------------
# 3. Matrice Laplacien 1D (Neumann)
# ------------------------------------------------------------
def build_laplacian_1D(N, ds):
    diag0 = -2.0 * np.ones(N+1)
    diag1 = np.ones(N)
    A = diags([diag1, diag0, diag1], offsets=[-1,0,1], format='csr')
    A[0, 1] = 2.0
    A[N, N-1] = 2.0
    return (D / ds**2) * A

Ax = build_laplacian_1D(Nx, dx)
Ay = build_laplacian_1D(Ny, dy)

Ix = eye(Nx+1, format='csr')
Iy = eye(Ny+1, format='csr')
A2D = kron(Iy, Ax) + kron(Ay, Ix)
Ntot = (Nx+1)*(Ny+1)

# ------------------------------------------------------------
# 4. Fonction reaction et flux
# ------------------------------------------------------------
def f(u):
    return u * (1.0 - u) * (u - a)

def G_vec(u_vec):
    return A2D @ u_vec + f(u_vec)

# ------------------------------------------------------------
# 5. Condition initiale (front plan vertical)
# ------------------------------------------------------------
x0 = Lx / 2.0
sigma = 2.0
U0 = 1.0 / (1.0 + np.exp((X - x0) / sigma))

# ------------------------------------------------------------
# 6. Pas de temps
# ------------------------------------------------------------
h = min(dx, dy)
mu_exp = 0.12
dt_exp = mu_exp * h**2 / D
Nt_exp = int(np.ceil(T / dt_exp))
dt_exp = T / Nt_exp

dt_imp = 10.0 * dt_exp
Nt_imp = int(np.ceil(T / dt_imp))
dt_imp = T / Nt_imp

print(f"h={h:.4f} | dt_exp={dt_exp:.6f} (Nt={Nt_exp}) | dt_imp={dt_imp:.4f} (Nt={Nt_imp})")

def front_position(U_mat, x, dx):
    jmid = Ny // 2
    u_mid = U_mat[jmid, :]
    idx = np.where((u_mid[:-1] >= 0.5) & (u_mid[1:] < 0.5))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return x[i] + dx * (0.5 - u_mid[i]) / (u_mid[i+1] - u_mid[i])

# Position theorique du front a t=T
c_th = np.sqrt(2.0*D) * (0.5 - a)
x_front_theo = Lx/2 + c_th * T
tol_convergence = 1e-1 # tolerance sur l'erreur de position

# ------------------------------------------------------------
# 7. Schema 1 : Runge-Kutta 4 explicite
# ------------------------------------------------------------
print("\n--- RK4 ---")
t_start = time.perf_counter()
U = U0.copy()
u_vec = U.reshape(-1)
front_RK4 = np.zeros(Nt_exp)

for n in range(Nt_exp):
    K1 = G_vec(u_vec)
    K2 = G_vec(u_vec + 0.5*dt_exp*K1)
    K3 = G_vec(u_vec + 0.5*dt_exp*K2)
    K4 = G_vec(u_vec + dt_exp*K3)
    u_vec = u_vec + (dt_exp/6.0)*(K1 + 2*K2 + 2*K3 + K4)
    U_mat = u_vec.reshape(Ny+1, Nx+1)
    front_RK4[n] = front_position(U_mat, x, dx)

U_RK4 = u_vec.reshape(Ny+1, Nx+1)
t_end = time.perf_counter()
t_RK4 = t_end - t_start

x_front_RK4 = front_RK4[-1] if not np.isnan(front_RK4[-1]) else np.nan
err_RK4 = abs(x_front_RK4 - x_front_theo) if not np.isnan(x_front_RK4) else np.nan
conv_RK4 = "OUI" if (not np.isnan(err_RK4) and err_RK4 < tol_convergence) else "NON (divergence ou erreur > tol)"

print(f"  Temps execution : {t_RK4:.3f} s")
print(f"  Position front final : {x_front_RK4:.3f} (theorique {x_front_theo:.3f})")
print(f"  Erreur : {err_RK4:.3e}")
print(f"  Convergence : {conv_RK4}")

# ------------------------------------------------------------
# 8. Schema 2 : Euler implicite ADI (Peaceman-Rachford)
# ------------------------------------------------------------
print("\n--- Euler implicite ADI ---")
t_start = time.perf_counter()

U_adi = U0.copy()
front_ADI = np.zeros(Nt_imp)

alpha_x = dt_imp / 2.0
alpha_y = dt_imp / 2.0
Mx_adi = eye(Nx+1) - alpha_x * Ax
My_adi = eye(Ny+1) - alpha_y * Ay

LUx = splu(Mx_adi.tocsc())
LUy = splu(My_adi.tocsc())

def solve_x(b):
    return LUx.solve(b)

def solve_y(b):
    return LUy.solve(b)

for n in range(Nt_imp):
    Uhalf = np.zeros_like(U_adi)
    # Demi-pas 1 : implicite en x
    for j in range(Ny+1):
        u_j = U_adi[j, :]
        Ay_u = Ay @ u_j.reshape(-1,1)
        rhs = u_j + alpha_y * Ay_u.ravel() + dt_imp * f(u_j)
        Uhalf_j = solve_x(rhs)
        Uhalf[j, :] = Uhalf_j
    # Demi-pas 2 : implicite en y
    U_new = np.zeros_like(Uhalf)
    for i in range(Nx+1):
        u_col = Uhalf[:, i]
        rhs = u_col + alpha_x * (Ax @ u_col)
        U_new_col = solve_y(rhs)
        U_new[:, i] = U_new_col
    U_adi = U_new
    front_ADI[n] = front_position(U_adi, x, dx)

U_ADI_final = U_adi.copy()
t_end = time.perf_counter()
t_ADI = t_end - t_start

x_front_ADI = front_ADI[-1] if not np.isnan(front_ADI[-1]) else np.nan
err_ADI = abs(x_front_ADI - x_front_theo) if not np.isnan(x_front_ADI) else np.nan
conv_ADI = "OUI" if (not np.isnan(err_ADI) and err_ADI < tol_convergence) else "NON"

print(f"  Temps execution : {t_ADI:.3f} s")
print(f"  Position front final : {x_front_ADI:.3f} (theorique {x_front_theo:.3f})")
print(f"  Erreur : {err_ADI:.3e}")
print(f"  Convergence : {conv_ADI}")

# ------------------------------------------------------------
# 9. Schema 3 : Adams-Bashforth 2 (explicite)
# ------------------------------------------------------------
print("\n--- Adams-Bashforth 2 ---")
t_start = time.perf_counter()

u_vec = U0.reshape(-1)
G_old = G_vec(u_vec)
u_vec = u_vec + dt_exp * G_old
G_new = G_vec(u_vec)
front_AB2 = np.zeros(Nt_exp)
front_AB2[0] = front_position(U0, x, dx)  # t=0
front_AB2[1] = front_position(u_vec.reshape(Ny+1,Nx+1), x, dx)

for n in range(2, Nt_exp):
    u_vec = u_vec + dt_exp * (1.5*G_new - 0.5*G_old)
    G_old = G_new
    G_new = G_vec(u_vec)
    front_AB2[n] = front_position(u_vec.reshape(Ny+1,Nx+1), x, dx)

U_AB2 = u_vec.reshape(Ny+1, Nx+1)
t_end = time.perf_counter()
t_AB2 = t_end - t_start

x_front_AB2 = front_AB2[-1] if not np.isnan(front_AB2[-1]) else np.nan
err_AB2 = abs(x_front_AB2 - x_front_theo) if not np.isnan(x_front_AB2) else np.nan
conv_AB2 = "OUI" if (not np.isnan(err_AB2) and err_AB2 < tol_convergence) else "NON"

print(f"  Temps execution : {t_AB2:.3f} s")
print(f"  Position front final : {x_front_AB2:.3f} (theorique {x_front_theo:.3f})")
print(f"  Erreur : {err_AB2:.3e}")
print(f"  Convergence : {conv_AB2}")

# ------------------------------------------------------------
# 10. Schema 4 : Adams-Moulton 2 (semi-implicite)
# ------------------------------------------------------------
print("\n--- Adams-Moulton 2 ---")
t_start = time.perf_counter()

M_AM2 = eye(Ntot) - (dt_imp/2.0) * A2D
LU_AM2 = splu(M_AM2.tocsc())

u_vec = U0.reshape(-1)
front_AM2 = np.zeros(Nt_imp)

for n in range(Nt_imp):
    rhs = u_vec + (dt_imp/2.0) * (A2D @ u_vec + 2.0 * f(u_vec))
    u_vec = LU_AM2.solve(rhs)
    U_mat = u_vec.reshape(Ny+1, Nx+1)
    front_AM2[n] = front_position(U_mat, x, dx)

U_AM2_final = u_vec.reshape(Ny+1, Nx+1)
t_end = time.perf_counter()
t_AM2 = t_end - t_start

x_front_AM2 = front_AM2[-1] if not np.isnan(front_AM2[-1]) else np.nan
err_AM2 = abs(x_front_AM2 - x_front_theo) if not np.isnan(x_front_AM2) else np.nan
conv_AM2 = "OUI" if (not np.isnan(err_AM2) and err_AM2 < tol_convergence) else "NON"

print(f"  Temps execution : {t_AM2:.3f} s")
print(f"  Position front final : {x_front_AM2:.3f} (theorique {x_front_theo:.3f})")
print(f"  Erreur : {err_AM2:.3e}")
print(f"  Convergence : {conv_AM2}")

# ------------------------------------------------------------
# 11. Resume final des performances et convergence
# ------------------------------------------------------------
print("\n" + "="*60)
print("RESUME FINAL (compilation termine)")
print("="*60)
print(f"{'Methode':<20} {'Temps (s)':<12} {'Erreur position':<16} {'Converge?'}")
print("-"*60)
print(f"{'RK4':<20} {t_RK4:<12.3f} {err_RK4:<16.3e} {conv_RK4}")
print(f"{'Euler implicite ADI':<20} {t_ADI:<12.3f} {err_ADI:<16.3e} {conv_ADI}")
print(f"{'AB2':<20} {t_AB2:<12.3f} {err_AB2:<16.3e} {conv_AB2}")
print(f"{'AM2':<20} {t_AM2:<12.3f} {err_AM2:<16.3e} {conv_AM2}")
print("="*60)

# ------------------------------------------------------------
# 12. VISUALISATIONS (figures)
# ------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -- Figure 1 : Profils 2D finaux (quatre sous-figures) --
fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
titles = ['RK4', 'Euler implicite ADI', 'Adams-Bashforth 2', 'Adams-Moulton 2']
data = [U_RK4, U_ADI_final, U_AB2, U_AM2_final]
for ax, dat, tit in zip(axes.flat, data, titles):
    im = ax.imshow(dat, extent=[0, Lx, 0, Ly], origin='lower',
                   cmap='jet', vmin=0, vmax=1)
    ax.set_title(tit)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
fig1.colorbar(im, ax=axes.ravel().tolist(), label='u')
fig1.suptitle('Profil u(x,y,T) – comparaison des schémas')
plt.tight_layout()

# -- Figure 2 : Isovaleurs u = 0.5 (position du front) --
plt.figure()
contour_args = {'levels': [0.5], 'linewidths': 2}
plt.contour(X, Y, U_RK4, colors='blue', **contour_args, label='RK4')
plt.contour(X, Y, U_ADI_final, colors='red', **contour_args, label='Euler impl. ADI')
plt.contour(X, Y, U_AB2, colors='green', **contour_args, label='AB2')
plt.contour(X, Y, U_AM2_final, colors='magenta', **contour_args, label='AM2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Isovaleur u=0.5 à T – position du front 2D')
plt.axis('equal')
plt.grid(True)

# -- Figure 3 : Suivi de la position du front au cours du temps --
t_exp = np.linspace(0, T, Nt_exp)
t_imp = np.linspace(0, T, Nt_imp)
plt.figure()
plt.plot(t_exp, front_RK4, 'b-', linewidth=1.5, label='RK4')
plt.plot(t_imp, front_ADI, 'r-', linewidth=1.5, label='Euler impl. ADI')
plt.plot(t_exp, front_AB2, 'g.', markersize=3, label='AB2')
plt.plot(t_imp, front_AM2, 'm-', linewidth=1.5, label='AM2')
# Position théorique du front (si vitesse constante)
x_front_theo = Lx/2 + c_th * np.linspace(0, T, 100)
plt.plot(np.linspace(0, T, 100), x_front_theo, 'k--',
         label=f'Théorique c* = {c_th:.3f}')
plt.xlabel('t')
plt.ylabel('x_front (y = Ly/2)')
plt.legend()
plt.grid(True)
plt.title('Position du front sur la ligne centrale')

# -- Figure 4 : Surface 3D du profil final RK4 --
fig4 = plt.figure()
ax3d = fig4.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(X, Y, U_RK4, cmap='jet', edgecolor='none')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('u')
ax3d.set_title('Profil final u(x,y,T) – RK4')
fig4.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)

plt.show()  # affiche toutes les figures
# ------------------------------------------------------------
# 13. Vérification de convergence (spatiale et temporelle)
# ------------------------------------------------------------
print("\n" + "="*60)
print("VERIFICATION DE CONVERGENCE (schéma AM2+ADI)")
print("="*60)

def erreur_L2(U, U_ref, dx, dy):
    """Calcule l'erreur L2 relative (ou absolue) entre deux grilles.
       U et U_ref doivent être aux mêmes positions."""
    return np.sqrt(np.sum((U - U_ref)**2) * dx * dy)

# Paramètres pour la convergence spatiale
Nx_list = [25, 50, 100, 200]
Ny_list = [25, 50, 100, 200]
# Grille de référence : plus fine (Nx_ref, Ny_ref)
Nx_ref, Ny_ref = 400, 400
dx_ref = Lx / Nx_ref
dy_ref = Ly / Ny_ref
# On calcule la solution de référence avec dt très petit
dt_ref = 0.005   # petit pas de temps
print(f"Calcul de la solution de référence avec Nx={Nx_ref}, dt={dt_ref}...")
# Construction des matrices pour la référence
Ax_ref = build_laplacian_1D(Nx_ref, dx_ref, D)
Ay_ref = build_laplacian_1D(Ny_ref, dy_ref, D)
alpha_ref = dt_ref / 2.0
Mx_ref = diags([1.0]*(Nx_ref+1), 0, format='csr') - alpha_ref * Ax_ref
My_ref = diags([1.0]*(Ny_ref+1), 0, format='csr') - alpha_ref * Ay_ref
LUx_ref = splu(Mx_ref.tocsc())
LUy_ref = splu(My_ref.tocsc())
# Grille de référence
x_ref = np.linspace(0, Lx, Nx_ref+1)
y_ref = np.linspace(0, Ly, Ny_ref+1)
X_ref, Y_ref = np.meshgrid(x_ref, y_ref)
U0_ref = 1.0 / (1.0 + np.exp((X_ref - x0) / sigma))
U_ref = U0_ref.copy()
Nt_ref = int(np.ceil(T / dt_ref))
dt_ref = T / Nt_ref
for n in range(Nt_ref):
    # AM2+ADI sur grille fine
    LapU = laplacien_2d(U_ref, dx_ref, dy_ref, D)
    RHS = U_ref + (dt_ref/2.0) * (LapU + 2.0*f(U_ref))
    # Demi-pas 1
    Ustar = np.zeros_like(RHS)
    for j in range(Ny_ref+1):
        Ustar[j, :] = LUx_ref.solve(RHS[j, :])
    # Demi-pas 2
    U_new = np.zeros_like(Ustar)
    for i in range(Nx_ref+1):
        U_new[:, i] = LUy_ref.solve(Ustar[:, i])
    U_ref = U_new
print("  Solution de référence calculée.")

# Convergence spatiale (temps fixe, maillage variable)
print("\n--- Convergence spatiale (erreur L2 à T) ---")
print("Nx     h        erreur L2    ordre")
err_prev = None
for Nx, Ny in zip(Nx_list, Ny_list):
    dx = Lx / Nx
    dy = Ly / Ny
    # Réinitialiser la grille courante
    x_cur = np.linspace(0, Lx, Nx+1)
    y_cur = np.linspace(0, Ly, Ny+1)
    X_cur, Y_cur = np.meshgrid(x_cur, y_cur)
    U_cur = 1.0 / (1.0 + np.exp((X_cur - x0) / sigma))
    # Matrices pour ce maillage
    Ax_cur = build_laplacian_1D(Nx, dx, D)
    Ay_cur = build_laplacian_1D(Ny, dy, D)
    alpha_cur = dt_imp / 2.0   # on utilise le dt_imp défini précédemment (pas trop petit)
    Mx_cur = diags([1.0]*(Nx+1), 0, format='csr') - alpha_cur * Ax_cur
    My_cur = diags([1.0]*(Ny+1), 0, format='csr') - alpha_cur * Ay_cur
    LUx_cur = splu(Mx_cur.tocsc())
    LUy_cur = splu(My_cur.tocsc())
    # Boucle temporelle identique à celle de AM2 (même Nt_imp)
    for n in range(Nt_imp):
        LapU = laplacien_2d(U_cur, dx, dy, D)
        RHS = U_cur + (dt_imp/2.0) * (LapU + 2.0*f(U_cur))
        Ustar = np.zeros_like(RHS)
        for j in range(Ny+1):
            Ustar[j, :] = LUx_cur.solve(RHS[j, :])
        U_new = np.zeros_like(Ustar)
        for i in range(Nx+1):
            U_new[:, i] = LUy_cur.solve(Ustar[:, i])
        U_cur = U_new
    # Interpolation de la solution de référence sur la grille courante
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((y_ref, x_ref), U_ref, bounds_error=False, fill_value=None)
    pts = np.array([(y, x) for y in y_cur for x in x_cur])
    U_ref_interp = interp(pts).reshape(Ny+1, Nx+1)
    err = erreur_L2(U_cur, U_ref_interp, dx, dy)
    if err_prev is None:
        order = np.nan
    else:
        order = np.log(err_prev / err) / np.log(2.0)  # raffinement facteur 2
    print(f"{Nx:3d}   {dx:.4f}   {err:.3e}   {order:.2f}")
    err_prev = err

# Convergence temporelle (maillage fixe, dt variable)
print("\n--- Convergence temporelle (erreur L2 à T) ---")
dt_list = [dt_imp, dt_imp/2, dt_imp/4, dt_imp/8]
print("dt         erreur L2    ordre")
err_prev = None
Nx_fixed, Ny_fixed = 100, 100
dx_fixed = Lx / Nx_fixed
dy_fixed = Ly / Ny_fixed
x_fixed = np.linspace(0, Lx, Nx_fixed+1)
y_fixed = np.linspace(0, Ly, Ny_fixed+1)
X_fixed, Y_fixed = np.meshgrid(x_fixed, y_fixed)
# Grille fixe, on précalcule les matrices
Ax_fixed = build_laplacian_1D(Nx_fixed, dx_fixed, D)
Ay_fixed = build_laplacian_1D(Ny_fixed, dy_fixed, D)
for dt in dt_list:
    Nt = int(np.ceil(T / dt))
    dt_cur = T / Nt
    alpha_cur = dt_cur / 2.0
    Mx_cur = diags([1.0]*(Nx_fixed+1), 0, format='csr') - alpha_cur * Ax_fixed
    My_cur = diags([1.0]*(Ny_fixed+1), 0, format='csr') - alpha_cur * Ay_fixed
    LUx_cur = splu(Mx_cur.tocsc())
    LUy_cur = splu(My_cur.tocsc())
    U_cur = 1.0 / (1.0 + np.exp((X_fixed - x0) / sigma))
    for n in range(Nt):
        LapU = laplacien_2d(U_cur, dx_fixed, dy_fixed, D)
        RHS = U_cur + (dt_cur/2.0) * (LapU + 2.0*f(U_cur))
        Ustar = np.zeros_like(RHS)
        for j in range(Ny_fixed+1):
            Ustar[j, :] = LUx_cur.solve(RHS[j, :])
        U_new = np.zeros_like(Ustar)
        for i in range(Nx_fixed+1):
            U_new[:, i] = LUy_cur.solve(Ustar[:, i])
        U_cur = U_new
    # Interpolation de la solution de référence sur la grille fixe
    interp = RegularGridInterpolator((y_ref, x_ref), U_ref, bounds_error=False, fill_value=None)
    pts = np.array([(y, x) for y in y_fixed for x in x_fixed])
    U_ref_interp = interp(pts).reshape(Ny_fixed+1, Nx_fixed+1)
    err = erreur_L2(U_cur, U_ref_interp, dx_fixed, dy_fixed)
    if err_prev is None:
        order = np.nan
    else:
        order = np.log(err_prev / err) / np.log(2.0)
    print(f"{dt_cur:.6f}   {err:.3e}   {order:.2f}")
    err_prev = err

print("\nVérification terminée.")
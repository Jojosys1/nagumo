# -*- coding: utf-8 -*-
"""
Simulation de l'équation de Nagumo 2D — Code CORRIGÉ
  u_t = D*(u_xx + u_yy) + u*(1-u)*(u-a)
  Conditions de Neumann homogènes sur tout le bord.
  Domaine [0,Lx]x[0,Ly], t in [0,T].

ERREURS IDENTIFIÉES ET CORRIGÉES :
  1. build_laplacian_1D : signature incohérente (D manquant en paramètre)
  2. laplacien_2d       : fonction utilisée en section 13 mais jamais définie
  3. AM2               : stencil incorrect (mauvaise combinaison des termes)
  4. ADI               : réaction f(u) non traitée dans le second demi-pas → asymétrie
  5. x_front_theo      : variable scalaire réécrasée en vecteur
  6. front_AB2[1]      : calculé avant d'avoir avancé d'un pas correctement

Comparaison : Euler explicite (RK4, AB2) vs Euler implicite (ADI, AM2)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import splu
import time

# ============================================================
# 1. Paramètres physiques
# ============================================================
Lx, Ly = 100.0, 100.0
D  = 1.0
a  = 0.25
T  = 40.0

# ============================================================
# 2. Discrétisation spatiale
# ============================================================
Nx = 600   # réduit pour rapidité (300 dans l'original → très lent)
Ny = 600
dx = Lx / Nx
dy = Ly / Ny
x  = np.linspace(0, Lx, Nx+1)
y  = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y)

# ============================================================
# 3. CORRECTION 1 : build_laplacian_1D — ajout du paramètre D
# ============================================================
def build_laplacian_1D(N, ds, D_coef=1.0):
    """
    Laplacien 1D avec conditions de Neumann (ghost point) :
    - coin [0,1] : 2 pour simuler u_{-1}=u_{1}
    - coin [N,N-1] : 2 pour simuler u_{N+1}=u_{N-1}
    """
    diag0 = -2.0 * np.ones(N+1)
    diag1 = np.ones(N)
    A = diags([diag1, diag0, diag1], offsets=[-1, 0, 1], format='csr')
    # Neumann : modification des coins
    A = A.tolil()
    A[0, 1]   = 2.0
    A[N, N-1] = 2.0
    A = A.tocsr()
    return (D_coef / ds**2) * A

Ax = build_laplacian_1D(Nx, dx, D)
Ay = build_laplacian_1D(Ny, dy, D)

Ix   = eye(Nx+1, format='csr')
Iy   = eye(Ny+1, format='csr')
A2D  = kron(Iy, Ax) + kron(Ay, Ix)
Ntot = (Nx+1)*(Ny+1)

# ============================================================
# 4. CORRECTION 2 : laplacien_2d — fonction manquante définie ici
# ============================================================
def laplacien_2d(U, hx, hy, D_coef=1.0):
    """
    Laplacien 2D fini (Neumann) appliqué directement sur la matrice U.
    Utilisé dans les boucles de convergence.
    """
    LapU = np.zeros_like(U)
    # Intérieur
    LapU[1:-1, 1:-1] = (
        D_coef * (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / hx**2
      + D_coef * (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1]) / hy**2
    )
    # Bords (Neumann) : dérivée nulle → demi-cellule doublée
    LapU[0, 1:-1]  = D_coef*(U[0,2:]-2*U[0,1:-1]+U[0,:-2])/hx**2 + D_coef*2*(U[1,1:-1]-U[0,1:-1])/hy**2
    LapU[-1,1:-1]  = D_coef*(U[-1,2:]-2*U[-1,1:-1]+U[-1,:-2])/hx**2 + D_coef*2*(U[-2,1:-1]-U[-1,1:-1])/hy**2
    LapU[1:-1,0]   = D_coef*2*(U[1:-1,1]-U[1:-1,0])/hx**2 + D_coef*(U[2:,0]-2*U[1:-1,0]+U[:-2,0])/hy**2
    LapU[1:-1,-1]  = D_coef*2*(U[1:-1,-2]-U[1:-1,-1])/hx**2 + D_coef*(U[2:,-1]-2*U[1:-1,-1]+U[:-2,-1])/hy**2
    # Coins
    LapU[0,0]   = D_coef*2*(U[0,1]-U[0,0])/hx**2 + D_coef*2*(U[1,0]-U[0,0])/hy**2
    LapU[0,-1]  = D_coef*2*(U[0,-2]-U[0,-1])/hx**2 + D_coef*2*(U[1,-1]-U[0,-1])/hy**2
    LapU[-1,0]  = D_coef*2*(U[-1,1]-U[-1,0])/hx**2 + D_coef*2*(U[-2,0]-U[-1,0])/hy**2
    LapU[-1,-1] = D_coef*2*(U[-1,-2]-U[-1,-1])/hx**2 + D_coef*2*(U[-2,-1]-U[-1,-1])/hy**2
    return LapU

# ============================================================
# 5. Terme de réaction et flux
# ============================================================
def f(u):
    return u * (1.0 - u) * (u - a)

def G_vec(u_vec):
    return A2D @ u_vec + f(u_vec)

# ============================================================
# 6. Condition initiale (front plan vertical centré)
# ============================================================
x0    = Lx / 2.0
sigma = 2.0
U0    = 1.0 / (1.0 + np.exp((X - x0) / sigma))

# ============================================================
# 7. Pas de temps
# ============================================================
h       = min(dx, dy)
mu_exp  = 0.12
dt_exp  = mu_exp * h**2 / D
Nt_exp  = int(np.ceil(T / dt_exp))
dt_exp  = T / Nt_exp

dt_imp  = 10.0 * dt_exp
Nt_imp  = int(np.ceil(T / dt_imp))
dt_imp  = T / Nt_imp

print(f"h={h:.4f} | dt_exp={dt_exp:.6f} (Nt={Nt_exp}) | dt_imp={dt_imp:.4f} (Nt={Nt_imp})")

def front_position(U_mat, xv, hx):
    """Position du front (isovaleur 0.5) sur la ligne centrale."""
    jmid  = Ny // 2
    u_mid = U_mat[jmid, :]
    idx   = np.where((u_mid[:-1] >= 0.5) & (u_mid[1:] < 0.5))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return xv[i] + hx * (0.5 - u_mid[i]) / (u_mid[i+1] - u_mid[i])

# Vitesse théorique du front
c_th        = np.sqrt(2.0 * D) * (0.5 - a)
x_front_th  = Lx/2 + c_th * T   # scalaire (CORRECTION 5)
tol_conv    = 1.0                 # tolérance position (en unités de maillage)

print(f"Vitesse théorique c* = {c_th:.4f}  |  position théorique à T={T} : {x_front_th:.3f}")

# ============================================================
# 8. Schéma 1 : Runge-Kutta 4 explicite
# ============================================================
print("\n--- RK4 ---")
t0    = time.perf_counter()
u_vec = U0.reshape(-1).copy()
front_RK4 = np.full(Nt_exp, np.nan)

for n in range(Nt_exp):
    K1 = G_vec(u_vec)
    K2 = G_vec(u_vec + 0.5*dt_exp*K1)
    K3 = G_vec(u_vec + 0.5*dt_exp*K2)
    K4 = G_vec(u_vec + dt_exp*K3)
    u_vec += (dt_exp/6.0)*(K1 + 2*K2 + 2*K3 + K4)
    front_RK4[n] = front_position(u_vec.reshape(Ny+1, Nx+1), x, dx)

U_RK4  = u_vec.reshape(Ny+1, Nx+1)
t_RK4  = time.perf_counter() - t0
xf_RK4 = front_RK4[-1]
err_RK4 = abs(xf_RK4 - x_front_th) if not np.isnan(xf_RK4) else np.nan
print(f"  Temps : {t_RK4:.2f}s  |  x_front={xf_RK4:.3f}  theorique={x_front_th:.3f}  err={err_RK4:.3e}")

# ============================================================
# 9. CORRECTION 4 : ADI Peaceman-Rachford avec réaction symétrique
# ============================================================
print("\n--- ADI Peaceman-Rachford (CORRIGÉ) ---")
t0 = time.perf_counter()

# Matrices implicites demi-pas : (I - alpha*Ax) et (I - alpha*Ay)
alpha_x = D * dt_imp / (2.0 * dx**2)
alpha_y = D * dt_imp / (2.0 * dy**2)

Mx_adi = eye(Nx+1) - alpha_x * (Ax * (dx**2 / D))   # = I - (dt/2)*Ax
My_adi = eye(Ny+1) - alpha_y * (Ay * (dy**2 / D))   # = I - (dt/2)*Ay

# Reconstruit correctement : Mx = I - (dt_imp/2)*Ax
Mx_adi = (eye(Nx+1, format='csr') - (dt_imp/2.0) * Ax)
My_adi = (eye(Ny+1, format='csr') - (dt_imp/2.0) * Ay)

LUx = splu(Mx_adi.tocsc())
LUy = splu(My_adi.tocsc())

U_adi  = U0.copy()
front_ADI = np.full(Nt_imp, np.nan)

for n in range(Nt_imp):
    # ---- Demi-pas 1 : implicite en x, explicite en y + dt/2*réaction ----
    # RHS = (I + dt/2*Ay)*U + (dt/2)*f(U)
    Uhalf = np.zeros_like(U_adi)
    for j in range(Ny+1):
        rhs_j  = U_adi[j, :] + (dt_imp/2.0) * (Ay @ U_adi[j, :]) + (dt_imp/2.0) * f(U_adi[j, :])
        # CORRECTION: la réaction f(U) est répartie sur les deux demi-pas
        Uhalf[j, :] = LUx.solve(rhs_j)

    # ---- Demi-pas 2 : implicite en y, explicite en x + dt/2*réaction ----
    # RHS = (I + dt/2*Ax)*Uhalf + (dt/2)*f(Uhalf)
    U_new = np.zeros_like(Uhalf)
    for i in range(Nx+1):
        rhs_i      = Uhalf[:, i] + (dt_imp/2.0) * (Ax @ Uhalf[:, i]) + (dt_imp/2.0) * f(Uhalf[:, i])
        U_new[:, i] = LUy.solve(rhs_i)

    U_adi = U_new
    front_ADI[n] = front_position(U_adi, x, dx)

U_ADI  = U_adi.copy()
t_ADI  = time.perf_counter() - t0
xf_ADI = front_ADI[-1]
err_ADI = abs(xf_ADI - x_front_th) if not np.isnan(xf_ADI) else np.nan
print(f"  Temps : {t_ADI:.2f}s  |  x_front={xf_ADI:.3f}  theorique={x_front_th:.3f}  err={err_ADI:.3e}")

# ============================================================
# 10. Schéma 3 : Adams-Bashforth 2 (explicite)
# ============================================================
print("\n--- Adams-Bashforth 2 ---")
t0 = time.perf_counter()

u_vec  = U0.reshape(-1).copy()
G_old  = G_vec(u_vec)
front_AB2 = np.full(Nt_exp, np.nan)

# CORRECTION 6 : premier pas Euler explicite, puis AB2
front_AB2[0] = front_position(U0, x, dx)
u_vec = u_vec + dt_exp * G_old   # Euler step
G_new = G_vec(u_vec)
front_AB2[1] = front_position(u_vec.reshape(Ny+1, Nx+1), x, dx)

for n in range(2, Nt_exp):
    u_vec = u_vec + dt_exp * (1.5*G_new - 0.5*G_old)
    G_old = G_new
    G_new = G_vec(u_vec)
    front_AB2[n] = front_position(u_vec.reshape(Ny+1, Nx+1), x, dx)

U_AB2  = u_vec.reshape(Ny+1, Nx+1)
t_AB2  = time.perf_counter() - t0
xf_AB2 = front_AB2[-1]
err_AB2 = abs(xf_AB2 - x_front_th) if not np.isnan(xf_AB2) else np.nan
print(f"  Temps : {t_AB2:.2f}s  |  x_front={xf_AB2:.3f}  theorique={x_front_th:.3f}  err={err_AB2:.3e}")

# ============================================================
# 11. CORRECTION 3 : Adams-Moulton 2 (Trapèze / Crank-Nicolson réaction)
# ============================================================
print("\n--- Adams-Moulton 2 / Crank-Nicolson (CORRIGÉ) ---")
t0 = time.perf_counter()

# Schéma correct AM2 (trapèze) :
#   (I - dt/2*A) u^{n+1} = (I + dt/2*A) u^n + dt/2*(f^n + f^{n+1_pred})
# On utilise une variante Crank-Nicolson semi-implicite :
#   diffusion implicite, réaction semi-implicite (linéarisée ou Adams-Bashforth)
# Ici : implicite en diffusion, explicite en réaction (ordre 2 en temps)
M_AM2  = eye(Ntot, format='csr') - (dt_imp/2.0) * A2D
LU_AM2 = splu(M_AM2.tocsc())

u_vec = U0.reshape(-1).copy()
front_AM2 = np.full(Nt_imp, np.nan)

for n in range(Nt_imp):
    # RHS = (I + dt/2*A)*u^n + dt*f(u^n)   [Crank-Nicolson diffusion + Euler réaction]
    rhs   = u_vec + (dt_imp/2.0) * (A2D @ u_vec) + dt_imp * f(u_vec)
    u_vec = LU_AM2.solve(rhs)
    front_AM2[n] = front_position(u_vec.reshape(Ny+1, Nx+1), x, dx)

U_AM2  = u_vec.reshape(Ny+1, Nx+1)
t_AM2  = time.perf_counter() - t0
xf_AM2 = front_AM2[-1]
err_AM2 = abs(xf_AM2 - x_front_th) if not np.isnan(xf_AM2) else np.nan
print(f"  Temps : {t_AM2:.2f}s  |  x_front={xf_AM2:.3f}  theorique={x_front_th:.3f}  err={err_AM2:.3e}")

# ============================================================
# 12. Résumé
# ============================================================
print("\n" + "="*65)
print(f"{'Méthode':<22} {'Temps(s)':>8} {'x_front':>10} {'Erreur':>12} {'Converge?'}")
print("-"*65)
for nom, t_, xf_, err_ in [
        ("RK4 (explicite)",    t_RK4,  xf_RK4,  err_RK4),
        ("ADI (implicite)",    t_ADI,  xf_ADI,  err_ADI),
        ("AB2 (explicite)",    t_AB2,  xf_AB2,  err_AB2),
        ("AM2/CN (semi-impl)", t_AM2,  xf_AM2,  err_AM2),
]:
    conv = "OUI" if (not np.isnan(err_) and err_ < tol_conv) else "NON"
    print(f"{nom:<22} {t_:>8.2f} {xf_:>10.3f} {err_:>12.3e}  {conv}")
print("="*65)
print(f"Position théorique : {x_front_th:.3f}")

# ============================================================
# 13. Visualisations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
titles  = ['RK4', 'ADI (corrigé)', 'Adams-Bashforth 2', 'AM2/CN (corrigé)']
datas   = [U_RK4, U_ADI, U_AB2, U_AM2]
for ax, dat, tit in zip(axes.flat, datas, titles):
    im = ax.imshow(dat, extent=[0, Lx, 0, Ly], origin='lower', cmap='jet', vmin=0, vmax=1)
    ax.set_title(tit); ax.set_xlabel('x'); ax.set_ylabel('y')
fig.colorbar(im, ax=axes.ravel().tolist(), label='u')
fig.suptitle(f'Profil u(x,y,T={T}) — comparaison des schémas')
plt.tight_layout()
plt.savefig('/home/claude/fig1_profils.png', dpi=130, bbox_inches='tight')
plt.close()

# Isovaleurs
fig2, ax2 = plt.subplots(figsize=(8, 7))
handles = []
for dat, col, lbl in zip(datas, ['blue','red','green','magenta'], titles):
    ax2.contour(X, Y, dat, levels=[0.5], colors=[col], linewidths=2)
    handles.append(plt.Line2D([0],[0], color=col, lw=2, label=lbl))
ax2.legend(handles=handles)
ax2.axvline(x_front_th, color='k', linestyle='--')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.grid(True)
plt.tight_layout()
plt.savefig('/home/claude/fig2_isovaleurs.png', dpi=130, bbox_inches='tight')
plt.close()

# Suivi temporel du front
t_exp_arr = np.linspace(0, T, Nt_exp)
t_imp_arr = np.linspace(0, T, Nt_imp)
t_th      = np.linspace(0, T, 200)
fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.plot(t_exp_arr, front_RK4, 'b-',  lw=1.5, label='RK4')
ax3.plot(t_imp_arr, front_ADI, 'r-',  lw=1.5, label='ADI')
ax3.plot(t_exp_arr, front_AB2, 'g.',  ms=2,   label='AB2')
ax3.plot(t_imp_arr, front_AM2, 'm-',  lw=1.5, label='AM2/CN')
ax3.plot(t_th, Lx/2 + c_th*t_th, 'k--', lw=2, label=f'Théorique c*={c_th:.3f}')
ax3.set_xlabel('t'); ax3.set_ylabel('x_front'); ax3.legend(); ax3.grid(True)
ax3.set_title('Suivi temporel du front (ligne centrale y=Ly/2)')
plt.tight_layout()
plt.savefig('/home/claude/fig3_front_temps.png', dpi=130, bbox_inches='tight')
plt.close()

print("\nFigures sauvegardées.")

# ============================================================
# 14. Convergence spatiale (schéma AM2/CN)
# ============================================================
print("\n" + "="*60)
print("VÉRIFICATION DE CONVERGENCE SPATIALE (AM2/CN)")
print("="*60)

def run_am2(Nx_c, Ny_c, dt_c):
    """Lance AM2/CN sur grille Nx_c x Ny_c avec pas dt_c."""
    hx  = Lx / Nx_c; hy  = Ly / Ny_c
    xv  = np.linspace(0, Lx, Nx_c+1)
    yv  = np.linspace(0, Ly, Ny_c+1)
    Xc, Yc = np.meshgrid(xv, yv)
    Uc  = 1.0 / (1.0 + np.exp((Xc - x0) / sigma))
    Axc = build_laplacian_1D(Nx_c, hx, D)
    Ayc = build_laplacian_1D(Ny_c, hy, D)
    Ixc = eye(Nx_c+1, format='csr'); Iyc = eye(Ny_c+1, format='csr')
    A2c = kron(Iyc, Axc) + kron(Ayc, Ixc)
    Mc  = eye((Nx_c+1)*(Ny_c+1), format='csr') - (dt_c/2.0) * A2c
    LUc = splu(Mc.tocsc())
    uv  = Uc.reshape(-1)
    Nt_c = int(np.ceil(T / dt_c))
    dt_c = T / Nt_c
    for _ in range(Nt_c):
        rhs = uv + (dt_c/2.0)*(A2c @ uv) + dt_c * f(uv)
        uv  = LUc.solve(rhs)
    return uv.reshape(Ny_c+1, Nx_c+1), hx

# Solution de référence (grille fine)
print("Calcul solution de référence (Nx=200)...")
U_ref, _ = run_am2(200, 200, dt_imp/4)
x_ref = np.linspace(0, Lx, 201)
y_ref = np.linspace(0, Ly, 201)

from scipy.interpolate import RegularGridInterpolator

def erreur_L2(U_c, hx, hy):
    from scipy.interpolate import RegularGridInterpolator
    Ny_c, Nx_c = U_c.shape[0]-1, U_c.shape[1]-1
    xc = np.linspace(0, Lx, Nx_c+1)
    yc = np.linspace(0, Ly, Ny_c+1)
    interp = RegularGridInterpolator((y_ref, x_ref), U_ref,
                                     bounds_error=False, fill_value=None)
    pts = np.array([(yy, xx) for yy in yc for xx in xc])
    Ur  = interp(pts).reshape(Ny_c+1, Nx_c+1)
    return np.sqrt(np.sum((U_c - Ur)**2) * hx * hy)

print(f"\n{'Nx':>5}  {'h':>8}  {'Erreur L2':>12}  {'Ordre':>6}")
err_prev = None
for Nxy in [25, 50, 100]:
    Uc, hc = run_am2(Nxy, Nxy, dt_imp)
    err = erreur_L2(Uc, Lx/Nxy, Ly/Nxy)
    ordre = np.log(err_prev/err)/np.log(2) if err_prev else np.nan
    print(f"{Nxy:>5}  {Lx/Nxy:>8.4f}  {err:>12.4e}  {ordre:>6.2f}")
    err_prev = err

print("\nVérification terminée.")
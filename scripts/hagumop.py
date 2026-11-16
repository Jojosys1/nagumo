import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

# =========================================================
# PARAMETRES PHYSIQUES
# =========================================================
Lx, Ly = 60, 60
D = 1
a = 0.25
T = 10000

# =========================================================
# DISCRETISATION SPATIALE
# =========================================================
Nx, Ny = 100, 100
dx, dy = Lx / Nx, Ly / Ny

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
X, Y = np.meshgrid(x, y)

Ntot = (Nx+1)*(Ny+1)

# =========================================================
# LAPLACIEN 1D (Neumann)
# =========================================================
def build_laplacian_1D(N, ds, D):
    diagonals = [np.ones(N), -2*np.ones(N+1), np.ones(N)]
    A = diags(diagonals, [-1, 0, 1], shape=(N+1, N+1)).tolil()

    # Neumann
    A[0,1] = 2
    A[-1,-2] = 2

    return (D / ds**2) * A.tocsc()

Ax = build_laplacian_1D(Nx, dx, D)
Ay = build_laplacian_1D(Ny, dy, D)

Ix = eye(Nx+1)
Iy = eye(Ny+1)

A2D = kron(Iy, Ax) + kron(Ay, Ix)

# =========================================================
# REACTION
# =========================================================
def f(u):
    return u * (1 - u) * (u - a)

# =========================================================
# CONDITION INITIALE
# =========================================================
x0 = Lx / 2
sigma = 2
U0 = 1 / (1 + np.exp((X - x0) / sigma))

# =========================================================
# PAS DE TEMPS
# =========================================================
h = min(dx, dy)

mu_exp = 0.12
dt_exp = mu_exp * h**2 / D
Nt_exp = int(np.ceil(T / dt_exp))
dt_exp = T / Nt_exp

dt_imp = 10 * dt_exp
Nt_imp = int(np.ceil(T / dt_imp))
dt_imp = T / Nt_imp

print(f"h={h:.4f} | dt_exp={dt_exp:.6f} | dt_imp={dt_imp:.4f}")

# =========================================================
# OUTILS
# =========================================================
def vec2mat(v):
    return v.reshape((Ny+1, Nx+1))

def front_position(U):
    u_mid = U[Ny//2, :]
    idx = np.where((u_mid[:-1] >= 0.5) & (u_mid[1:] < 0.5))[0]
    if len(idx) > 0:
        i = idx[0]
        return x[i] + dx*(0.5 - u_mid[i])/(u_mid[i+1]-u_mid[i])
    return 0

# =========================================================
# SCHEMA RK4
# =========================================================
u_vec = U0.flatten()

def G(u):
    return A2D @ u + f(u)

front_RK4 = []

for _ in range(Nt_exp):
    K1 = G(u_vec)
    K2 = G(u_vec + 0.5*dt_exp*K1)
    K3 = G(u_vec + 0.5*dt_exp*K2)
    K4 = G(u_vec + dt_exp*K3)

    u_vec += (dt_exp/6)*(K1 + 2*K2 + 2*K3 + K4)

    front_RK4.append(front_position(vec2mat(u_vec)))

U_RK4 = vec2mat(u_vec)

# =========================================================
# EULER IMPLICITE
# =========================================================
I2D = eye(Ntot)
M_EI = (I2D - dt_imp*A2D).tocsc()
solver_EI = splu(M_EI)

u_vec = U0.flatten()
front_EI = []

for _ in range(Nt_imp):
    rhs = u_vec + dt_imp * f(u_vec)
    u_vec = solver_EI.solve(rhs)
    front_EI.append(front_position(vec2mat(u_vec)))

U_EI = vec2mat(u_vec)

# =========================================================
# ADAMS-BASHFORTH 2
# =========================================================
u_vec = U0.flatten()
G_old = G(u_vec)

u_vec += dt_exp * G_old
G_new = G(u_vec)

front_AB2 = [0]

for _ in range(1, Nt_exp):
    u_vec += dt_exp*(1.5*G_new - 0.5*G_old)
    G_old = G_new
    G_new = G(u_vec)
    front_AB2.append(front_position(vec2mat(u_vec)))

U_AB2 = vec2mat(u_vec)

# =========================================================
# ADAMS-MOULTON 2
# =========================================================
M_AM2 = (I2D - dt_imp/2 * A2D).tocsc()
solver_AM2 = splu(M_AM2)

u_vec = U0.flatten()
front_AM2 = []

for _ in range(Nt_imp):
    rhs = u_vec + (dt_imp/2)*(A2D @ u_vec + 2*f(u_vec))
    u_vec = solver_AM2.solve(rhs)
    front_AM2.append(front_position(vec2mat(u_vec)))

U_AM2 = vec2mat(u_vec)

# =========================================================
# VITESSE THEORIQUE
# =========================================================
c_th = np.sqrt(2*D)*(0.5 - a)
print("c* =", c_th)

# =========================================================
# FIGURES
# =========================================================
plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(U_RK4, extent=[0,Lx,0,Ly], origin='lower')
plt.title("RK4"); plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(U_EI, extent=[0,Lx,0,Ly], origin='lower')
plt.title("Euler implicite"); plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(U_AB2, extent=[0,Lx,0,Ly], origin='lower')
plt.title("AB2"); plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(U_AM2, extent=[0,Lx,0,Ly], origin='lower')
plt.title("AM2"); plt.colorbar()

plt.suptitle("Comparaison des schémas")
plt.show()

# Front en temps
t_exp = np.linspace(0, T, Nt_exp)
t_imp = np.linspace(0, T, Nt_imp)

plt.figure()
plt.plot(t_exp, front_RK4, label="RK4")
plt.plot(t_imp, front_EI, label="Euler implicite")
plt.plot(t_exp, front_AB2, '.', label="AB2")
plt.plot(t_imp, front_AM2, label="AM2")

plt.plot([0,T], [Lx/2, Lx/2 + c_th*T], '--', label="Théorique")

plt.legend()
plt.xlabel("t")
plt.ylabel("position du front")
plt.grid()
plt.show()

# Surface 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U_RK4)
ax.set_title("Surface 3D RK4")
plt.show()
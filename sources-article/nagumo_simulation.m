function nagumo_simulation
% NAGUMO_SIMULATION
% Simulation 1D de l'equation de Nagumo:
%   u_t = d*u_xx + u(1-u)(u-a)
% avec conditions de Neumann homogenes.

clear; clc; close all;

%% Parametres physiques et numeriques
p.L  = 60.0;
p.N  = 1200;
p.d  = 1.0;
p.a  = 0.25;
p.T  = 25.0;
p.x0 = 20.0;
p.sigma = 1.5;

p.dx = p.L / p.N;
p.x  = linspace(0, p.L, p.N + 1)';

% CFL explicite diffusion (regle conservative)
dt_cfl = 0.5 * p.dx^2 / p.d;

% Pas de temps par methode
dt.euler = 0.40 * dt_cfl;
dt.rk4   = 0.60 * dt_cfl;
dt.ab2   = 0.35 * dt_cfl;
dt.heun  = 0.45 * dt_cfl;

% Solution initiale: front lisse
u0 = 0.5 * (1.0 - tanh((p.x - p.x0) / p.sigma));

%% Execution des methodes
resEuler = run_solver('euler', u0, dt.euler, p);
resRK4   = run_solver('rk4',   u0, dt.rk4,   p);
resAB2   = run_solver('ab2',   u0, dt.ab2,   p);
resHeun  = run_solver('heun',  u0, dt.heun,  p);

%% Vitesse theorique et numerique
c_th = sqrt(2 * p.d) * (0.5 - p.a);
fprintf('Vitesse theorique c_th = %.6f\n', c_th);
fprintf('Euler : c_num = %.6f\n', resEuler.c_num);
fprintf('RK4   : c_num = %.6f\n', resRK4.c_num);
fprintf('AB2   : c_num = %.6f\n', resAB2.c_num);
fprintf('Heun  : c_num = %.6f\n', resHeun.c_num);

%% Figure 1: profils finaux
figure('Name', 'Profils finaux', 'Color', 'w');
plot(p.x, resEuler.u, 'LineWidth', 1.2); hold on;
plot(p.x, resRK4.u,   'LineWidth', 1.2);
plot(p.x, resAB2.u,   'LineWidth', 1.2);
plot(p.x, resHeun.u,  'LineWidth', 1.2);
legend('Euler', 'RK4', 'AB2', 'Heun', 'Location', 'best');
xlabel('x'); ylabel('u(x,T)');
title('Equation de Nagumo: profils finaux');
grid on;

%% Figure 2: position du front x_{0.5}(t)
figure('Name', 'Trajectoire du front', 'Color', 'w');
plot(resEuler.t, resEuler.xhalf, 'LineWidth', 1.2); hold on;
plot(resRK4.t,   resRK4.xhalf,   'LineWidth', 1.2);
plot(resAB2.t,   resAB2.xhalf,   'LineWidth', 1.2);
plot(resHeun.t,  resHeun.xhalf,  'LineWidth', 1.2);
legend('Euler', 'RK4', 'AB2', 'Heun', 'Location', 'best');
xlabel('t'); ylabel('x_{0.5}(t)');
title('Suivi du front u=0.5');
grid on;

%% Figure 3: comparaison vitesse numerique vs theorique
figure('Name', 'Comparaison des vitesses', 'Color', 'w');
labels = categorical({'Euler','RK4','AB2','Heun','Theorique'});
vals = [resEuler.c_num, resRK4.c_num, resAB2.c_num, resHeun.c_num, c_th];
bar(labels, vals);
ylabel('Vitesse');
title('Vitesse du front: numerique vs theorique');
grid on;

% Figure 4: energie discrete au cours du temps
figure('Name', 'Energie discrete', 'Color', 'w');
plot(resEuler.t, resEuler.energy, 'LineWidth', 1.2); hold on;
plot(resRK4.t,   resRK4.energy,   'LineWidth', 1.2);
plot(resAB2.t,   resAB2.energy,   'LineWidth', 1.2);
plot(resHeun.t,  resHeun.energy,  'LineWidth', 1.2);
legend('Euler', 'RK4', 'AB2', 'Heun', 'Location', 'best');
xlabel('t'); ylabel('E_h(t)');
title('Decroissance de l''energie discrete');
grid on;

%% Test de convergence en temps (sur RK4 par defaut)
conv = time_convergence_study(u0, p, dt.rk4);
figure('Name', 'Convergence temporelle RK4', 'Color', 'w');
loglog(conv.dt, conv.err, '-o', 'LineWidth', 1.2); hold on;
if numel(conv.dt) >= 2
    ref = conv.err(1) * (conv.dt / conv.dt(1)).^4;
    loglog(conv.dt, ref, '--', 'LineWidth', 1.2);
    legend('Erreur L2', 'Pente 4 reference', 'Location', 'best');
else
    legend('Erreur L2', 'Location', 'best');
end
xlabel('\Deltat'); ylabel('Erreur L2 a T');
title('Etude de convergence en temps (RK4)');
grid on;

%% Animation optionnelle (decommenter si besoin)
% animate_solution(resRK4.snapshots, resRK4.snap_t, p.x);

end

function out = run_solver(method, u0, dt, p)
Nt = ceil(p.T / dt);
dt = p.T / Nt;

t = (0:Nt)' * dt;
u = u0;

% stockage front
xhalf = zeros(Nt + 1, 1);
xhalf(1) = front_position(u, p.x, 0.5);

% snapshots pour animation
nskip = max(1, floor(Nt / 150));
snapshots = zeros(numel(u0), floor(Nt / nskip) + 1);
snap_t = zeros(floor(Nt / nskip) + 1, 1);
energy = zeros(Nt + 1, 1);
is = 1;
snapshots(:, is) = u;
snap_t(is) = 0.0;
energy(1) = discrete_energy(u, p);

phi_prev = [];
for n = 1:Nt
    switch lower(method)
        case 'euler'
            phi = rhs(u, p);
            u = u + dt * phi;

        case 'rk4'
            k1 = rhs(u, p);
            k2 = rhs(u + 0.5 * dt * k1, p);
            k3 = rhs(u + 0.5 * dt * k2, p);
            k4 = rhs(u + dt * k3, p);
            u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);

        case 'ab2'
            phi = rhs(u, p);
            if n == 1
                % demarrage AB2 par un pas RK4
                k1 = phi;
                k2 = rhs(u + 0.5 * dt * k1, p);
                k3 = rhs(u + 0.5 * dt * k2, p);
                k4 = rhs(u + dt * k3, p);
                u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);
            else
                u = u + 0.5 * dt * (3.0 * phi - phi_prev);
            end
            phi_prev = phi;

        case 'heun'
            phi_n = rhs(u, p);
            up = u + dt * phi_n;
            phi_p = rhs(up, p);
            u = u + 0.5 * dt * (phi_n + phi_p);

        otherwise
            error('Methode inconnue: %s', method);
    end

    % projection douce sur [0,1] pour limiter les erreurs numeriques locales
    u = min(max(u, -0.05), 1.05);

    xhalf(n + 1) = front_position(u, p.x, 0.5);
    energy(n + 1) = discrete_energy(u, p);

    if mod(n, nskip) == 0
        is = is + 1;
        snapshots(:, is) = u;
        snap_t(is) = t(n + 1);
    end
end

% estimation vitesse sur la moitie finale du signal
k1 = floor(0.55 * (Nt + 1));
k2 = Nt + 1;
coef = polyfit(t(k1:k2), xhalf(k1:k2), 1);
c_num = coef(1);

out.t = t;
out.u = u;
out.xhalf = xhalf;
out.c_num = c_num;
out.snapshots = snapshots(:, 1:is);
out.snap_t = snap_t(1:is);
out.energy = energy;
end

function du = rhs(u, p)
uxx = lap_neumann(u, p.dx);
fu = u .* (1.0 - u) .* (u - p.a);
du = p.d * uxx + fu;
end

function uxx = lap_neumann(u, dx)
% points fantomes pour Neumann homogene
um = [u(2); u; u(end-1)];
uxx = (um(3:end) - 2.0 * um(2:end-1) + um(1:end-2)) / dx^2;
end

function xh = front_position(u, x, level)
% cherche la premiere traversee descendante du niveau "level"
idx = find((u(1:end-1) - level) .* (u(2:end) - level) <= 0, 1, 'first');
if isempty(idx)
    xh = NaN;
    return;
end
u1 = u(idx); u2 = u(idx + 1);
x1 = x(idx); x2 = x(idx + 1);
if abs(u2 - u1) < 1e-14
    xh = 0.5 * (x1 + x2);
else
    xh = x1 + (level - u1) * (x2 - x1) / (u2 - u1);
end
end

function conv = time_convergence_study(u0, p, dt_base)
% Compare plusieurs pas de temps a une reference fine RK4
nLevels = 4;
dt_list = dt_base ./ (2.^(0:nLevels-1));

% reference fine
res_ref = run_solver('rk4', u0, dt_list(end) / 2, p);
u_ref = res_ref.u;

err = zeros(size(dt_list));
for k = 1:numel(dt_list)
    res = run_solver('rk4', u0, dt_list(k), p);
    err(k) = sqrt(p.dx * sum((res.u - u_ref).^2));
end

conv.dt = dt_list(:);
conv.err = err(:);
end

function animate_solution(U, t, x)
figure('Name', 'Animation Nagumo', 'Color', 'w');
for k = 1:size(U, 2)
    plot(x, U(:, k), 'LineWidth', 1.4);
    ylim([-0.1, 1.1]);
    xlabel('x'); ylabel('u(x,t)');
    title(sprintf('t = %.3f', t(k)));
    grid on;
    drawnow;
end
end

function E = discrete_energy(u, p)
ux = gradient(u, p.dx);
F = -(u.^4) / 4.0 + (1.0 + p.a) * (u.^3) / 3.0 - p.a * (u.^2) / 2.0;
E = p.dx * sum(0.5 * p.d * ux.^2 - F);
end

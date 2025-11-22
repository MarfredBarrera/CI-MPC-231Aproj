import time as pytime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import custom_vjp
from tqdm import tqdm


# ==========================================
# 0. CONFIGURATION (Tuned for Hopping)
# ==========================================
class Config:
    m = 1.0
    g = 9.81
    dt = 0.025
    rho = 2.0

    u_max = 100.0       # Increased to ensure we have power to jump high
    u_min = -100.0

    # Weights
    W_x = jnp.array([100.0, 5.0]) # Stronger position incentive
    W_u = jnp.array([1e-3])       # Cheaper control to encourage aggressive jumps
    W_air = 1000.0                # Stronger "Get off the floor" penalty

    # Horizon
    N_horizon = 50      # 1.25 seconds (Critical fix: allows seeing the landing)
    max_iter_warm = 100 # More time to find the initial gait
    max_iter_mpc = 2
    sim_duration = 100

cfg = Config()

# ==========================================
# 1. DYNAMICS (Exact Parity + Mask Shift)
# ==========================================

def get_solver_terms(y, v, u):
    P = (1.0 / cfg.m)
    v_pred = v + (-cfg.g) * cfg.dt
    ci = v_pred
    ci_drift = ci + (y / cfg.dt)
    return P, ci_drift, v_pred

@custom_vjp
def solve_contact_impulse(y, v, u):
    P, ci, _ = get_solver_terms(y, v, u)
    ci_total = ci + (u / cfg.m) * cfg.dt
    lamb = -ci_total / P
    lamb = jnp.maximum(lamb, 0.0)
    return lamb

def solve_contact_impulse_fwd(y, v, u):
    lam = solve_contact_impulse(y, v, u)
    return lam, (lam, y, v, u)

def solve_contact_impulse_bwd(res, g_lam):
    lam, y, v, u = res
    P, ci, _ = get_solver_terms(y, v, u)

    D = 1.0 / (lam**2 + 1e-6)
    scale = -1.0 / (P + cfg.rho * D)

    db_dv = 1.0
    db_dy = 1.0 / cfg.dt
    db_du = (1.0 / cfg.m) * cfg.dt

    grad_y = g_lam * scale * db_dy
    grad_v = g_lam * scale * db_dv
    grad_u = g_lam * scale * db_du
    return grad_y, grad_v, grad_u

solve_contact_impulse.defvjp(solve_contact_impulse_fwd, solve_contact_impulse_bwd)

def step_cimpc(state, u):
    y, v = state
    u = jnp.squeeze(u)
    u = jnp.clip(u, cfg.u_min, cfg.u_max)

    # --- Tune: Shift mask to ensure 100% force at y=0 ---
    # Sigmoid centered at y=0.05 means at y=0 it is ~0.92 (near full power)
    contact_mask = jax.nn.sigmoid(-50.0 * (y - 0.05))
    u_effective = u * contact_mask

    lam = solve_contact_impulse(y, v, u_effective)

    v_next = v + (-cfg.g + u_effective/cfg.m) * cfg.dt + (lam / cfg.m)
    y_next = y + v_next * cfg.dt
    return jnp.array([y_next, v_next])

# ==========================================
# 2. COSTS
# ==========================================

def running_cost(state, u, x_ref, u_ref, weight_air):
    x_err = state - x_ref
    u_err = u - u_ref
    c_reg = jnp.sum(cfg.W_x * x_err**2) + jnp.sum(cfg.W_u * u_err**2)

    # Air Time Cost
    prob_contact = jax.nn.sigmoid(-50.0 * state[0])
    c_air = weight_air * prob_contact
    return c_reg + c_air

def terminal_cost(state, x_ref):
    x_err = state - x_ref
    return jnp.sum((cfg.W_x * 10.0) * x_err**2)

# ==========================================
# 3. FDDP SOLVER
# ==========================================

def compute_defects(x_traj, u_traj):
    def single_defect(x_curr, u_curr, x_next):
        x_pred = step_cimpc(x_curr, u_curr)
        return x_pred - x_next
    return jax.vmap(single_defect)(x_traj[:-1], u_traj, x_traj[1:])

def get_derivatives(x_traj, u_traj, x_refs, u_refs, air_weights):
    def step_derivs(x, u, xr, ur, w_air):
        fx, fu = jax.jacrev(step_cimpc, (0, 1))(x, u)
        cost_fn = lambda _x, _u: running_cost(_x, _u, xr, ur, w_air)
        lx, lu = jax.grad(cost_fn, (0, 1))(x, u)
        lxx = jax.hessian(cost_fn, 0)(x, u)
        luu = jax.hessian(cost_fn, 1)(x, u)
        lxu = jax.jacfwd(jax.grad(cost_fn, 0), 1)(x, u)
        fu = jnp.reshape(fu, (2, 1))
        lu = jnp.reshape(lu, (1,))
        luu = jnp.reshape(luu, (1, 1))
        lxu = jnp.reshape(lxu, (2, 1))
        return fx, fu, lx, lu, lxx, luu, lxu
    return jax.vmap(step_derivs)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights)

def backward_pass(derivatives, x_traj, defects, reg_mu=1e-3):
    fx, fu, lx, lu, lxx, luu, lxu = derivatives
    N = fx.shape[0]
    V_x = jax.grad(terminal_cost)(x_traj[-1], x_traj[-1])
    V_xx = jax.hessian(terminal_cost)(x_traj[-1], x_traj[-1])
    ks = jnp.zeros((N, 1))
    Ks = jnp.zeros((N, 1, 2))
    def loop_body(carry, i):
        V_x, V_xx = carry
        V_x_plus = V_x + V_xx @ defects[i]
        Q_x = lx[i] + fx[i].T @ V_x_plus
        Q_u = lu[i] + fu[i].T @ V_x_plus
        Q_xx = lxx[i] + fx[i].T @ V_xx @ fx[i]
        Q_uu = luu[i] + fu[i].T @ V_xx @ fu[i]
        Q_ux = lxu[i].T + fu[i].T @ V_xx @ fx[i]
        Q_uu_reg = Q_uu + reg_mu * jnp.eye(1)
        k = -jnp.linalg.solve(Q_uu_reg, Q_u)
        K = -jnp.linalg.solve(Q_uu_reg, Q_ux)
        V_x_prev = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        V_xx_prev = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        return (V_x_prev, V_xx_prev), (k, K)
    _, (ks, Ks) = jax.lax.scan(loop_body, (V_x, V_xx), jnp.arange(N)[::-1])
    return ks[::-1], Ks[::-1]

def forward_pass_fddp(x0, x_traj, u_traj, defects, ks, Ks, alpha):
    def scan_fn(carry, inputs):
        x_curr = carry
        i, u_bar, x_bar, k, K, defect = inputs
        dx = x_curr - x_bar
        u_new = u_bar + alpha * k + K @ dx
        u_new = jnp.clip(u_new, cfg.u_min, cfg.u_max)
        x_pred = step_cimpc(x_curr, u_new)
        x_next = x_pred - (1.0 - alpha) * defect
        return x_next, (x_next, u_new)
    inputs = (jnp.arange(u_traj.shape[0]), u_traj, x_traj[:-1], ks, Ks, defects)
    _, (x_new_traj, u_new_traj) = jax.lax.scan(scan_fn, x0, inputs)
    return jnp.vstack([x0, x_new_traj]), u_new_traj

def total_cost(x_traj, u_traj, x_refs, u_refs, air_weights):
    rc = jax.vmap(running_cost)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights)
    tc = terminal_cost(x_traj[-1], x_refs[-1])
    return jnp.sum(rc) + tc

def solve_fddp(x0, x_traj_guess, u_traj_guess, x_refs, u_refs, air_weights, max_iter):
    x_traj, u_traj = x_traj_guess, u_traj_guess
    for i in range(max_iter):
        defects = compute_defects(x_traj, u_traj)
        derivs = get_derivatives(x_traj, u_traj, x_refs, u_refs, air_weights)
        ks, Ks = backward_pass(derivs, x_traj, defects)
        best_cost = total_cost(x_traj, u_traj, x_refs, u_refs, air_weights)
        best_x, best_u = x_traj, u_traj
        for alpha in [1.0, 0.8, 0.5, 0.2]:
            xn, un = forward_pass_fddp(x0, x_traj, u_traj, defects, ks, Ks, alpha)
            cn = total_cost(xn, un, x_refs, u_refs, air_weights)
            if cn < best_cost:
                best_cost = cn
                best_x, best_u = xn, un
                break
        x_traj, u_traj = best_x, best_u
    return x_traj, u_traj

# ==========================================
# 4. SIMULATION
# ==========================================

def run_mpc_simulation():
    N = cfg.N_horizon
    x_current = jnp.array([1.0, 0.0])
    u_traj = jnp.zeros((N,))

    def rollout_pure(x, u):
        xn = step_cimpc(x, u)
        return xn, xn
    _, x_traj = jax.lax.scan(rollout_pure, x_current, u_traj)
    x_traj = jnp.vstack([x_current, x_traj])

    x_refs = jnp.tile(jnp.array([1.5, 0.0]), (N+1, 1))
    u_refs = jnp.tile(jnp.array([0.0]), (N, 1))
    air_weights = jnp.ones((N,)) * cfg.W_air

    print(f"Warm Start ({cfg.max_iter_warm} iters)...")
    x_traj, u_traj = solve_fddp(x_current, x_traj, u_traj, x_refs, u_refs, air_weights, cfg.max_iter_warm)

    sim_data = {'x': [x_current], 'u': [], 'cost': []}

    print("Simulating...")
    start_time = pytime.time()
    # Run for 200 steps (5 seconds) to see multiple hops
    for t in tqdm(range(200)):
        x_traj, u_traj = solve_fddp(x_current, x_traj, u_traj, x_refs, u_refs, air_weights, cfg.max_iter_mpc)

        u_applied = u_traj[0]
        x_next = step_cimpc(x_current, u_applied)

        sim_data['x'].append(x_next)
        sim_data['u'].append(jnp.array(u_applied).flatten())
        sim_data['cost'].append(total_cost(x_traj, u_traj, x_refs, u_refs, air_weights))

        # Shift
        u_traj = jnp.roll(u_traj, -1, axis=0).at[-1].set(0.0)
        x_traj = jnp.roll(x_traj, -1, axis=0)
        x_term_guess = step_cimpc(x_traj[-2], 0.0)
        x_traj = x_traj.at[-1].set(x_term_guess)
        x_current = x_next
        x_traj = x_traj.at[0].set(x_current)

    print(f"Freq: {200/(pytime.time()-start_time):.2f} Hz")
    return sim_data

# ==========================================
# 5. PLOTTING
# ==========================================
data = run_mpc_simulation()
x_hist = jnp.array(data['x'])
u_hist = jnp.array(data['u']).flatten()
t_axis = jnp.arange(len(x_hist)) * cfg.dt

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.title("Stabilized Pogo Hopping (Horizon N=50)")
plt.plot(t_axis, x_hist[:, 0], label="Height")
plt.axhline(1.5, color='r', linestyle='--', label="Target")
plt.axhline(0.0, color='k', linewidth=2)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.step(t_axis[:-1], u_hist, where='post', color='orange', label="Control")
plt.ylabel("Force (N)")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(t_axis[:-1], data['cost'], color='purple', label="Cost")
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot.png', dpi=300)

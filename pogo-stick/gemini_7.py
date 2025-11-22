import time as pytime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import custom_vjp
from tqdm import tqdm


# ==========================================
# 0. CONFIGURATION (Parity Updated)
# ==========================================
class Config:
    m = 1.0
    g = 9.81
    dt = 0.025
    rho = 2.0

    u_max = 100.0
    u_min = -100.0

    W_x = jnp.array([100.0, 5.0])
    W_u = jnp.array([1e-3])
    W_air = 2000.0       # Increased for discrete activation

    N_horizon = 50
    max_iter_warm = 50
    max_iter_mpc = 2

    # FDDP Merit Function Weights
    mu_defect = 1000.0   # Weight for physical violation (gaps)

cfg = Config()

# ==========================================
# 1. DYNAMICS (Exact Parity)
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

    # Forward pass clipping (still needed for safety)
    u = jnp.clip(u, cfg.u_min, cfg.u_max)

    contact_mask = jax.nn.sigmoid(-50.0 * (y - 0.05))
    u_effective = u * contact_mask

    lam = solve_contact_impulse(y, v, u_effective)
    v_next = v + (-cfg.g + u_effective/cfg.m) * cfg.dt + (lam / cfg.m)
    y_next = y + v_next * cfg.dt
    return jnp.array([y_next, v_next])

# ==========================================
# 2. COSTS (Paper Parity: Discrete Air Time)
# ==========================================

def total_cost(x_traj, u_traj, x_refs, u_refs, air_weights):
    rc = jax.vmap(running_cost)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights)
    tc = terminal_cost(x_traj[-1], x_refs[-1])
    return jnp.sum(rc) + tc

def running_cost(state, u, x_ref, u_ref, weight_air):
    x_err = state - x_ref
    u_err = u - u_ref
    c_reg = jnp.sum(cfg.W_x * x_err**2) + jnp.sum(cfg.W_u * u_err**2)

    # Discrepancy Fix #2: Use the discrete weight passed in
    # The 'weight_air' is now a binary-like switch (0 or W_air) computed outside
    # We only penalize if the foot is actually low (state[0] < 0.05)
    is_ground = jax.nn.sigmoid(-50.0 * (state[0] - 0.05))
    c_air = weight_air * is_ground * (state[0]**2) # Penalize height error near ground

    return c_reg + c_air

def terminal_cost(state, x_ref):
    x_err = state - x_ref
    return jnp.sum((cfg.W_x * 10.0) * x_err**2)

# ==========================================
# 3. FDDP SOLVER (Paper Parity: Box-DDP + Merit Function)
# ==========================================

def compute_defects(x_traj, u_traj):
    def single_defect(x_curr, u_curr, x_next):
        return step_cimpc(x_curr, u_curr) - x_next
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

def backward_pass(derivatives, x_traj, u_traj, defects, reg_mu=1e-3):
    fx, fu, lx, lu, lxx, luu, lxu = derivatives
    N = fx.shape[0]
    V_x = jax.grad(terminal_cost)(x_traj[-1], x_traj[-1])
    V_xx = jax.hessian(terminal_cost)(x_traj[-1], x_traj[-1])
    ks = jnp.zeros((N, 1))
    Ks = jnp.zeros((N, 1, 2))

    def loop_body(carry, i):
        V_x, V_xx, u_nom = carry # Need nominal u for Box-DDP
        V_x_plus = V_x + V_xx @ defects[i]

        Q_x = lx[i] + fx[i].T @ V_x_plus
        Q_u = lu[i] + fu[i].T @ V_x_plus
        Q_xx = lxx[i] + fx[i].T @ V_xx @ fx[i]
        Q_uu = luu[i] + fu[i].T @ V_xx @ fu[i]
        Q_ux = lxu[i].T + fu[i].T @ V_xx @ fx[i]

        # --- Discrepancy Fix #3: Box-DDP Projection (Simplified) ---
        Q_uu_reg = Q_uu + reg_mu * jnp.eye(1)

        # Unconstrained gains
        k_open = -jnp.linalg.solve(Q_uu_reg, Q_u)
        K_open = -jnp.linalg.solve(Q_uu_reg, Q_ux)

        # Projected Newton Check:
        # If u_nom + k hits limit, we clamp k and zero out K (feedback)
        # This is a "Quasi-Box" approach sufficient for 1D
        u_prospective = u_nom + k_open
        is_clamped = (u_prospective > cfg.u_max) | (u_prospective < cfg.u_min)

        # If clamped: k puts us exactly at limit, K becomes zero (cannot feedback)
        k_clamped = jnp.clip(u_prospective, cfg.u_min, cfg.u_max) - u_nom

        # Select based on constraint activation
        # Note: For 1D u, this scalar selection works. For vector u, need active set algebra.
        k = jnp.where(is_clamped, k_clamped, k_open)
        K = jnp.where(is_clamped, 0.0 * K_open, K_open) # Kill feedback on active constraint

        V_x_prev = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        V_xx_prev = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        return (V_x_prev, V_xx_prev, u_traj[i-1]), (k, K)

    # Pass u_traj shifted by -1 to align indices in reverse scan
    u_traj_shifted = jnp.roll(u_traj, 1, axis=0)
    _, (ks, Ks) = jax.lax.scan(loop_body, (V_x, V_xx, u_traj[-1]), jnp.arange(N)[::-1])
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

def calculate_merit(x_traj, u_traj, defects, x_refs, u_refs, air_weights):
    """
    Discrepancy Fix #1: FDDP Merit Function
    Phi = Cost + mu * ||defects||_1
    """
    rc = jax.vmap(running_cost)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights)
    tc = terminal_cost(x_traj[-1], x_refs[-1])
    cost_total = jnp.sum(rc) + tc

    # L1 norm of defects
    infeasibility = jnp.sum(jnp.abs(defects))

    return cost_total + cfg.mu_defect * infeasibility, cost_total

def solve_fddp(x0, x_traj_guess, u_traj_guess, x_refs, u_refs, air_weights, max_iter):
    x_traj, u_traj = x_traj_guess, u_traj_guess

    for i in range(max_iter):
        defects = compute_defects(x_traj, u_traj)
        # Calculate initial Merit
        merit_current, cost_current = calculate_merit(x_traj, u_traj, defects, x_refs, u_refs, air_weights)

        derivs = get_derivatives(x_traj, u_traj, x_refs, u_refs, air_weights)
        ks, Ks = backward_pass(derivs, x_traj, u_traj, defects)

        best_merit = merit_current
        best_x, best_u = x_traj, u_traj

        # Line Search on MERIT, not just Cost
        for alpha in [1.0, 0.8, 0.5, 0.2, 0.05]:
            xn, un = forward_pass_fddp(x0, x_traj, u_traj, defects, ks, Ks, alpha)
            defects_new = compute_defects(xn, un)
            merit_new, cost_new = calculate_merit(xn, un, defects_new, x_refs, u_refs, air_weights)

            # Goldstein-like acceptance (simplified to monotonic reduction)
            if merit_new < best_merit:
                best_merit = merit_new
                best_x, best_u = xn, un
                break # Found a step that improves merit

        x_traj, u_traj = best_x, best_u
    return x_traj, u_traj

# ==========================================
# 4. SIMULATION (Paper Parity: History-based Air Cost)
# ==========================================

def run_mpc_simulation():
    N = cfg.N_horizon
    x_current = jnp.array([1.0, 0.0])
    u_traj = jnp.zeros((N,))

    def rollout_pure(x, u): return step_cimpc(x, u), step_cimpc(x, u)
    _, x_traj = jax.lax.scan(rollout_pure, x_current, u_traj)
    x_traj = jnp.vstack([x_current, x_traj])

    x_refs = jnp.tile(jnp.array([1.5, 0.0]), (N+1, 1))
    u_refs = jnp.tile(jnp.array([0.0]), (N, 1))

    # --- History State for Air Time Cost ---
    # "Weights... activated only when swing leg time surpasses threshold"
    time_on_ground = 0
    threshold_time = 12 # 0.3s / 0.025s = 12 steps

    sim_data = {'x': [x_current], 'u': [], 'cost': []}
    print("Simulating...")
    start_time = pytime.time()

    for t in tqdm(range(150)):
        # 1. Update Air Time Logic
        # Check if robot is "loitering" (y near 0)
        if x_current[0] < 0.05:
            time_on_ground += 1
        else:
            time_on_ground = 0

        # If loitering > threshold, activate weights
        current_air_weight = cfg.W_air if time_on_ground > threshold_time else 0.0
        air_weights = jnp.ones((N,)) * current_air_weight

        # 2. Solve
        x_traj, u_traj = solve_fddp(x_current, x_traj, u_traj, x_refs, u_refs, air_weights, cfg.max_iter_mpc)

        # 3. Apply
        u_applied = u_traj[0]
        x_next = step_cimpc(x_current, u_applied)

        # Log
        sim_data['x'].append(x_next)
        sim_data['u'].append(jnp.array(u_applied).flatten())
        sim_data['cost'].append(total_cost(x_traj, u_traj, x_refs, u_refs, air_weights))

        # 4. Shift (With Parity Logic)
        u_traj = jnp.roll(u_traj, -1, axis=0).at[-1].set(0.0)
        x_traj = jnp.roll(x_traj, -1, axis=0)
        x_term_guess = step_cimpc(x_traj[-2], 0.0)
        x_traj = x_traj.at[-1].set(x_term_guess)
        x_current = x_next
        x_traj = x_traj.at[0].set(x_current)

    print(f"Freq: {150/(pytime.time()-start_time):.2f} Hz")
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
plt.title("Pogo Hopping (Full Algorithm Parity)")
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
plt.savefig("pogo_7.png", dpi=500)

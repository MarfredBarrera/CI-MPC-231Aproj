import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from jax import custom_vjp
from functools import partial
import numpy as np

# ==========================================
# 0. CONFIGURATION (2D System)
# ==========================================
class Config:
    # Dimensions
    n_q = 3 # [x, y, theta]
    n_v = 3 # [dx, dy, dtheta]
    n_x = 6 # [q, v]
    n_u = 2 # [Torque, Force]
    # Contact: [Normal, Tangent+, Tangent-]
    n_c = 3 

    # Physics Parameters
    m = 5.0     # Mass (kg)
    I = 1.0     # Inertia
    l = 1.0     # Leg Length
    g = 9.81
    dt = 0.02
    
    # Solver Parameters
    rho = 1.0 # Constraint relaxation (smooths gradients)
    
    # Input Limits
    u_max = jnp.array([50.0, 1500.0]) # [Torque, Force]
    u_min = jnp.array([-50.0, 0.0])   # Force cannot pull
    
    # Weights for MPC
    # State: [x, y, th, dx, dy, dth]
    # High penalty on Height(y) and Angle(th). Lower on horizontal pos(x).
    W_x = jnp.array([0.0, 100.0, 100.0, 10.0, 10.0, 1.0]) 
    W_u = jnp.array([0.5, 0.001]) # Cheap to use force, expensive to use torque
    W_air = 5000.0       # Cost to enforce foot clearance during flight phases

    N_horizon = 30
    max_iter_mpc = 15
    mu_defect = 500.0

cfg = Config()

# ==========================================
# 1. DYNAMICS (Verified 2D Rigid Leg)
# ==========================================

def get_foot_kinematics(q, q_dot):
    """Calculates foot position and velocity."""
    x, y, th = q
    vx, vy, w = q_dot
    s, c = jnp.sin(th), jnp.cos(th)
    
    # p_foot = hip + vector
    p_foot = jnp.array([x + cfg.l*s, y - cfg.l*c])
    
    # v_foot = v_hip + w x r
    v_foot = jnp.array([vx + cfg.l*c*w, vy + cfg.l*s*w])
    return p_foot, v_foot

def compute_common_terms(q, q_dot, u):
    """
    Computes M, B, J, h for the n-dimensional system.
    """
    x, y, th = q
    
    # --- Mass Matrix (n_v, n_v) ---
    M = jnp.diag(jnp.array([cfg.m, cfg.m, cfg.I]))
    M_inv = jnp.diag(jnp.array([1.0/cfg.m, 1.0/cfg.m, 1.0/cfg.I]))
    
    # --- Bias (Gravity) ---
    h = jnp.array([0.0, cfg.m * cfg.g, 0.0])
    
    # --- Input Matrix B (n_v, n_u) ---
    # Depends on contact mask to prevent "rocket boot"
    p_foot, _ = get_foot_kinematics(q, q_dot)
    
    # Soft mask: 1.0 if on ground, 0.0 if in air
    contact_mask = jax.nn.sigmoid(-100.0 * (p_foot[1] - 0.05))
    
    s, c = jnp.sin(th), jnp.cos(th)
    
    # Col 0: Torque (affects Theta)
    # Col 1: Force (affects X, Y via leg vector), masked by contact
    B = jnp.array([
        [0.0, -s * contact_mask],
        [0.0,  c * contact_mask],
        [1.0,  0.0]
    ])
    
    # --- Contact Jacobian J (n_c, n_v) ---
    # Rows: [Normal, Tangent+, Tangent-]
    # Tangents are masked so we don't have air-friction
    J = jnp.vstack([
        jnp.array([0.0, 1.0, cfg.l * s]),               # Normal
        jnp.array([1.0, 0.0, cfg.l * c]) * contact_mask, # Tangent +
        jnp.array([-1.0, 0.0, -cfg.l * c]) * contact_mask # Tangent -
    ])
    
    # --- Unconstrained Motion Terms (Eq 11) ---
    # b = J * M_inv * ( (-h + Bu)*dt + M*q_dot )
    # Note: added M*q_dot (momentum) because we step velocity: v_next = v + dv
    
    impulse_forces = (-h + B @ u) * cfg.dt
    vel_star = q_dot + M_inv @ impulse_forces
    
    b = J @ vel_star
    
    # Baumgarte Stabilization (phi / dt)
    # Only applied to Normal dimension [0]
    phi = p_foot[1] # Height
    b = b.at[0].set(b[0] + phi / cfg.dt)
    
    # Delassus Matrix: A = J * M_inv * J^T
    A = J @ M_inv @ J.T
    
    return A, b, M_inv, J, phi

def solve_lcp_pgs(A, b, max_iter=50):
    """
    Solves LCP: 0 <= lambda _|_ A*lambda + b >= 0
    Generalized Projected Gauss-Seidel for n-dimensions.
    """
    n = b.shape[0]
    lambda_init = jnp.zeros_like(b)
    
    def iteration(lam, _):
        def row_update(lam_curr, i):
            sigma = jnp.dot(A[i], lam_curr) - A[i, i] * lam_curr[i]
            val = -(b[i] + sigma) / (A[i, i] + 1e-6)
            return lam_curr.at[i].set(jnp.maximum(0.0, val)), None
            
        lam_next, _ = jax.lax.scan(row_update, lam, jnp.arange(n))
        return lam_next, None

    lambda_sol, _ = jax.lax.scan(iteration, lambda_init, None, length=max_iter)
    return lambda_sol

@custom_vjp
def solve_contact_impulse(q, q_dot, u):
    A, b, _, _, _ = compute_common_terms(q, q_dot, u)
    return solve_lcp_pgs(A, b)

def solve_contact_impulse_fwd(q, q_dot, u):
    lambda_c = solve_contact_impulse(q, q_dot, u)
    return lambda_c, (lambda_c, q, q_dot, u)

def solve_contact_impulse_bwd(res, g_lambda):
    lambda_c, q, q_dot, u = res
    (A, b, M_inv, J, phi), vjp_fn = jax.vjp(compute_common_terms, q, q_dot, u)
    
    # Relaxed Complementarity for Gradients
    lam_sq = lambda_c**2 + 1e-6
    D_mat = jnp.diag(1.0 / lam_sq)
    H_relaxed = A + cfg.rho * D_mat
    
    # Implicit Differentiation
    # H * dlambda + db = 0  => dlambda = -H_inv * db
    nu = -jnp.linalg.solve(H_relaxed, g_lambda)
    
    grad_b = nu
    grad_A = jnp.outer(nu, lambda_c)
    
    grad_q, grad_q_dot, grad_u = vjp_fn((grad_A, grad_b, jnp.zeros_like(M_inv), jnp.zeros_like(J), jnp.zeros_like(phi)))
    return grad_q, grad_q_dot, grad_u

solve_contact_impulse.defvjp(solve_contact_impulse_fwd, solve_contact_impulse_bwd)

def step_cimpc(state, u):
    """
    General n-dimensional Dynamics Step
    """
    q = state[:cfg.n_q]
    q_dot = state[cfg.n_q:]

    u = jnp.clip(u, cfg.u_min, cfg.u_max)
    
    # 1. Contact Impulse
    lambda_c = solve_contact_impulse(q, q_dot, u)
    
    # 2. Integration
    # We must recompute matrices to apply the impulse
    # (In optimized C++ code we would pass these through, but JAX creates clean graphs this way)
    A, b, M_inv, J, phi = compute_common_terms(q, q_dot, u)
    
    # Note: We need B again. 
    p_foot, _ = get_foot_kinematics(q, q_dot)
    contact_mask = jax.nn.sigmoid(-100.0 * (p_foot[1] - 0.05))
    s, c = jnp.sin(q[2]), jnp.cos(q[2])
    B = jnp.array([[0.0, -s*contact_mask], [0.0, c*contact_mask], [1.0, 0.0]])
    h = jnp.array([0.0, cfg.m*cfg.g, 0.0])
    
    # v_next = v + M_inv * ( (-h + Bu)*dt + J^T * lambda )
    forces = (-h + B @ u) * cfg.dt
    contact_impulse = J.T @ lambda_c
    
    q_dot_next = q_dot + M_inv @ (forces + contact_impulse)
    
    # q_next = q + v_next * dt
    q_next = q + q_dot_next * cfg.dt
    
    return jnp.concatenate([q_next, q_dot_next])

# ==========================================
# 2. COSTS
# ==========================================

def total_cost(x_traj, u_traj, x_refs, u_refs, air_weights):
    rc = jax.vmap(running_cost)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights)
    tc = terminal_cost(x_traj[-1], x_refs[-1])
    return jnp.sum(rc) + tc

def running_cost(state, u, x_ref, u_ref, weight_air):
    x_err = state - x_ref
    u_err = u - u_ref
    
    c_reg = jnp.sum(cfg.W_x * x_err**2) + jnp.sum(cfg.W_u * u_err**2)

    # Air Cost: Penalize low foot height during flight phases
    q, q_dot = state[:3], state[3:]
    p_foot, _ = get_foot_kinematics(q, q_dot)
    foot_h = p_foot[1]
    
    # If weight_air > 0, we want foot_h to be large.
    # Cost = weight * sigmoid(low) * (1/height) or just penalize being close to 0
    is_ground = jax.nn.sigmoid(-50.0 * (foot_h - 0.05))
    c_air = weight_air * is_ground * (1.0 - foot_h)**2 
    
    return c_reg + c_air

def terminal_cost(state, x_ref):
    x_err = state - x_ref
    return jnp.sum((cfg.W_x * 10.0) * x_err**2)

# ==========================================
# 3. FDDP SOLVER (Generalized)
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
        
        return fx, fu, lx, lu, lxx, luu, lxu
    return jax.vmap(step_derivs)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights)

def backward_pass(derivatives, x_traj, u_traj, defects, reg_mu=1e-3):
    fx, fu, lx, lu, lxx, luu, lxu = derivatives
    N = fx.shape[0]
    
    V_x = jax.grad(terminal_cost)(x_traj[-1], x_traj[-1])
    V_xx = jax.hessian(terminal_cost)(x_traj[-1], x_traj[-1])
    
    # Initialize ks and Ks with correct dimensions
    # ks: (N, n_u), Ks: (N, n_u, n_x)
    
    def loop_body(carry, i):
        V_x, V_xx, u_nom = carry
        V_x_plus = V_x + V_xx @ defects[i]

        Q_x = lx[i] + fx[i].T @ V_x_plus
        Q_u = lu[i] + fu[i].T @ V_x_plus
        Q_xx = lxx[i] + fx[i].T @ V_xx @ fx[i]
        Q_uu = luu[i] + fu[i].T @ V_xx @ fu[i]
        Q_ux = lxu[i].T + fu[i].T @ V_xx @ fx[i]

        # Regularization (Generalized to n_u)
        Q_uu_reg = Q_uu + reg_mu * jnp.eye(cfg.n_u)

        # Gains
        k_open = -jnp.linalg.solve(Q_uu_reg, Q_u)
        K_open = -jnp.linalg.solve(Q_uu_reg, Q_ux)

        # Box-DDP
        u_prospective = u_nom + k_open
        k_clamped = jnp.clip(u_prospective, cfg.u_min, cfg.u_max) - u_nom
        
        # Check clamping per-channel
        is_clamped = (u_prospective > cfg.u_max) | (u_prospective < cfg.u_min)
        
        # If any channel is clamped, use clamped k, else open k.
        # Note: True Box-DDP is more complex (active set), this is a soft approx.
        k = k_clamped 
        
        # Mask Feedback for clamped dims (broadcasting)
        # If channel j is clamped, K[j, :] should be 0.
        # is_clamped shape is (n_u,) -> reshape to (n_u, 1)
        mask = 1.0 - is_clamped.astype(jnp.float32).reshape(-1, 1)
        K = K_open * mask

        V_x_prev = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        V_xx_prev = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        
        return (V_x_prev, V_xx_prev, u_traj[i-1]), (k, K)

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
    _, (x_new, u_new) = jax.lax.scan(scan_fn, x0, inputs)
    return jnp.vstack([x0, x_new]), u_new

def solve_fddp(x0, x_traj, u_traj, x_refs, u_refs, air_weights, max_iter):
    for _ in range(max_iter):
        defects = compute_defects(x_traj, u_traj)
        
        # JIT-friendly "current merit" calculation
        # We need to compute costs to compare
        merit_curr = jnp.sum(jax.vmap(running_cost)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights))
        
        derivs = get_derivatives(x_traj, u_traj, x_refs, u_refs, air_weights)
        ks, Ks = backward_pass(derivs, x_traj, u_traj, defects)

        best_merit = merit_curr
        best_x, best_u = x_traj, u_traj

        # Line Search
        for alpha in [1.0, 0.5, 0.1]:
            xn, un = forward_pass_fddp(x0, x_traj, u_traj, defects, ks, Ks, alpha)
            merit_new = jnp.sum(jax.vmap(running_cost)(xn[:-1], un, x_refs[:-1], u_refs, air_weights))
            
            # Acceptance logic (tracer safe)
            improved = merit_new < best_merit
            best_merit = jnp.where(improved, merit_new, best_merit)
            best_x = jnp.where(improved, xn, best_x)
            best_u = jnp.where(improved, un, best_u)

        x_traj, u_traj = best_x, best_u
    return x_traj, u_traj

# ==========================================
# 4. SIMULATION LOOP
# ==========================================

def run_mpc_simulation(N_steps):
    # Initial State: [x, y, th, dx, dy, dth]
    x_current = jnp.array([0.0, 1.2, 0.0, 
                           0.5, 0.0, 0.0])
    
    # Init Trajectories
    u_traj = jnp.zeros((cfg.N_horizon, cfg.n_u))
    def rollout_pure(x, u): return step_cimpc(x, u), step_cimpc(x, u)
    _, x_traj = jax.lax.scan(rollout_pure, x_current, u_traj)
    x_traj = jnp.vstack([x_current, x_traj])

    # Reference Trajectory Construction
    # Goal: Move Right (v_x = 1.0) at Height (y = 1.3)
    target_vel = 1.0
    target_h = 1.3
    
    init_carry = (x_current, x_traj, u_traj, 0.0)
    
    def sim_step(carry, t):
        x_curr, x_tr, u_tr, tog = carry
        
        # 1. Build Horizon Reference
        # Ref X moves away from current X at target velocity
        ref_x = x_curr[0] + target_vel * jnp.arange(cfg.N_horizon + 1) * cfg.dt
        ref_y = jnp.ones(cfg.N_horizon + 1) * target_h
        ref_th = jnp.zeros(cfg.N_horizon + 1)
        
        # Combine into state reference
        # [x, y, th, dx, dy, dth]
        x_refs = jnp.vstack([
            ref_x, ref_y, ref_th,
            jnp.ones(cfg.N_horizon+1)*target_vel, 
            jnp.zeros(cfg.N_horizon+1), 
            jnp.zeros(cfg.N_horizon+1)
        ]).T
        
        u_refs = jnp.zeros((cfg.N_horizon, cfg.n_u))
        
        # 2. Flight Phase Heuristic
        # If we have been on ground > X steps, encourage flight cost
        q, q_dot = x_curr[:3], x_curr[3:]
        p_foot, _ = get_foot_kinematics(q, q_dot)
        is_ground = p_foot[1] < 0.05
        tog_next = jnp.where(is_ground, tog + 1.0, 0.0)
        
        # If on ground for > 0.2s (10 steps), activate air weight
        w_air = jnp.where(tog_next > 10.0, cfg.W_air, 0.0)
        air_weights = jnp.ones(cfg.N_horizon) * w_air
        
        # 3. Solve MPC
        x_opt, u_opt = solve_fddp(x_curr, x_tr, u_tr, x_refs, u_refs, air_weights, cfg.max_iter_mpc)
        
        # 4. Step Physics
        u_applied = u_opt[0]
        x_next = step_cimpc(x_curr, u_applied)
        
        # 5. Shift Warmstart
        u_shift = jnp.roll(u_opt, -1, axis=0).at[-1].set(0.0)
        x_shift = jnp.roll(x_opt, -1, axis=0)
        # Naive terminal guess
        x_shift = x_shift.at[-1].set(step_cimpc(x_shift[-2], jnp.zeros(cfg.n_u)))
        x_shift = x_shift.at[0].set(x_next)
        
        return (x_next, x_shift, u_shift, tog_next), (x_next, u_applied)

    _, (x_hist, u_hist) = jax.lax.scan(sim_step, init_carry, jnp.arange(N_steps))
    return x_hist, u_hist



def animator(x_data, u_data, filename="2D_hopper.gif"):
    # Ensure data is numpy
    x_data = np.array(x_data)
    u_data = np.array(u_data)
    
    # Create figure with 3 subplots using correct layout setup
    # height_ratios gives the animation (top) more space than the plots (bottom)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})

    # --- Top Plot: Animation ---
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ground_line = ax.axhline(0, color='black', linewidth=2)
    body_circle = Circle((0, 0), 0.15, color='cornflowerblue', zorder=10)
    ax.add_patch(body_circle)
    leg_line, = ax.plot([], [], 'k-', linewidth=4, zorder=5)
    foot_point, = ax.plot([], [], 'ro', markersize=6, zorder=6)
    trail, = ax.plot([], [], 'b:', linewidth=1, alpha=0.5)
    
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    force_text = ax.text(0.05, 0.90, '', transform=ax.transAxes)

    # --- Bottom Plot: Inputs ---
    ax2.set_xlim(0, len(u_data)*cfg.dt)
    
    # Dynamic Y limits with padding
    f_min, f_max = np.min(u_data[:,1]), np.max(u_data[:,1])
    y1_pad = (f_max - f_min) * 0.1 if f_max != f_min else 1.0
    ax2.set_ylim(f_min - y1_pad, f_max + y1_pad)

    t_min, t_max = np.min(u_data[:,0]), np.max(u_data[:,0])
    y2_pad = (t_max - t_min) * 0.1 if t_max != t_min else 1.0
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force Input (N)')
    ax2.grid(True, linestyle='--', alpha=0.3)

    ax3 = ax2.twinx()
    ax3.set_ylabel('Torque Input (Nm)')
    ax3.set_ylim(t_min - y2_pad, t_max + y2_pad)
    
    force_line, = ax2.plot([], [], 'b-', label='Force (N)')
    torque_line, = ax3.plot([], [], 'r-', label='Torque (Nm)')
    
    # Combine legends from both axes
    lines = [force_line, torque_line]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper right')

    def get_leg_coords(state):
        x, y, th = state[:3]
        foot_x = x + cfg.l * np.sin(th)
        foot_y = y - cfg.l * np.cos(th)
        return (x, y), (foot_x, foot_y)

    def update(frame):
        state = x_data[frame]
        u = u_data[frame]
        
        (hx, hy), (fx, fy) = get_leg_coords(state)
        
        body_circle.center = (hx, hy)
        leg_line.set_data([hx, fx], [hy, fy])
        
        # Color leg red when applying force
        force_intensity = np.clip(u[1] / 1000.0, 0, 1)
        leg_line.set_color((force_intensity, 0, 0))
        
        foot_point.set_data([fx], [fy])
        
        trail_len = 50
        start_idx = 0
        # start_idx = max(0, frame - trail_len) # uncomment to limit trail length

        trail.set_data(x_data[start_idx:frame+1, 0], x_data[start_idx:frame+1, 1])
        
        # Camera Tracking
        ax.set_xlim([min(x_data[:,0])-1.0, max(x_data[:,0])+1.0])
        ax.set_ylim([min(x_data[:,1])-1.0, max(x_data[:,1])+1.0])

        time_text.set_text(f"Time: {frame * cfg.dt:.2f} s")
        force_text.set_text(f"Force: {u[1]:.1f} N | Torque: {u[0]:.1f} Nm")

        # Update Plot Lines
        current_time = np.arange(frame+1) * cfg.dt
        torque_line.set_data(current_time, u_data[:frame+1, 0])
        force_line.set_data(current_time, u_data[:frame+1, 1])
        
        return body_circle, leg_line, foot_point, trail, time_text, force_text, torque_line, force_line

    # Prevent overlap
    plt.tight_layout()

    ms_per_frame = int(cfg.dt * 1000)
    ani = animation.FuncAnimation(fig, update, frames=len(x_data)-1, 
                                  init_func=None, blit=False, interval=ms_per_frame)
    
    print(f"Saving animation to {filename}...")
    ani.save("./animations/"+ filename, writer='pillow', fps=30)
    print("Done.")

# ==========================================
# 5. EXECUTION & PLOT
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='2D Hopper CI-MPC Simulation')
    parser.add_argument('--load', type=str, help='Load data from file instead of running simulation')
    parser.add_argument('--save', type=str, default='sim_data.npz', help='Save simulation data to file')
    parser.add_argument('--steps', type=int, default=150, help='Number of simulation steps')
    parser.add_argument('--no-animate', action='store_true', help='Skip animation generation')
    parser.add_argument('--output', type=str, default='2D_hopper.gif', help='Animation output filename')
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing data
        print(f"Loading data from {args.load}...")
        data = np.load(args.load)
        x_data = data['x']
        u_data = data['u']
        print(f"Loaded: x_data shape {x_data.shape}, u_data shape {u_data.shape}")
    else:
        # Run simulation
        print("Compiling and Running 2D CI-MPC...")
        run_jit = jax.jit(run_mpc_simulation, static_argnums=(0,))
        x_data, u_data = run_jit(args.steps)
        
        # Save data
        print(f"Saving simulation data to {args.save}...")
        np.savez(args.save, 
                 x=np.array(x_data), 
                 u=np.array(u_data))
        print(f"Data saved!")
    
    t = jnp.arange(len(x_data)) * cfg.dt

    if not args.no_animate:
        print("Generating animation...")
        animator(x_data, u_data, filename=args.output)
        print(f"Animation saved to {args.output}")
    else:
        print("Skipping animation (--no-animate flag set)")
    
    # # Plot 1: Trajectory
    # plt.subplot(3, 1, 1)
    # plt.title("2D Hopper MPC Trajectory")
    # plt.plot(x_data[:, 0], x_data[:, 1], 'b-', label='Hip Path')
    # plt.axhline(0, color='k', linewidth=2)
    # plt.ylabel("Height (m)")
    # plt.legend()
    # plt.grid(True)
    
    # # Plot 2: States
    # plt.subplot(3, 1, 2)
    # plt.plot(t, x_data[:, 2], label="Theta (rad)")
    # plt.plot(t, x_data[:, 3], label="Vel X (m/s)")
    # plt.axhline(1.0, color='r', linestyle='--', label="Target Vel")
    # plt.legend()
    # plt.grid(True)
    
    # # Plot 3: Controls
    # plt.subplot(3, 1, 3)
    # plt.plot(t, u_data[:, 1], 'r', label="Force (N)")
    # plt.plot(t, u_data[:, 0], 'g', label="Torque (Nm)")
    # plt.xlabel("Time (s)")
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig("cimpc_2d_result.png")
    # print("Done. Saved plot to cimpc_2d_result.png")
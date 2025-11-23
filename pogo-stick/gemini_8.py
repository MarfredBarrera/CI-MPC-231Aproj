import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import custom_vjp
from functools import partial
# ==========================================
# 0. CONFIGURATION
# ==========================================
class Config:
    # System Dimensions:
    # n_q = 1 (configuration dimension, height)
    # n_v = 1 (velocity dimension)
    # n_x = 2 (state dimension: [q, v])
    # n_u = 1 (control dimension)
    # n_c = 1 (contact dimension)

    m = jax.numpy.array(1.0)  # Scalar mass
    g = 9.81
    dt = 0.025
    rho = 2.0 # constraint relaxation parameter used in gradient of contact forces (lambda)
    B_fn = lambda self, phi: (jax.nn.sigmoid(-100.0 * phi)+0.01) * jnp.eye(1)
    # Have B be a function of state (phi = height from ground). However, for convergence purposes, give the controller a small amt (1%) of control to prevent it from getting stuck.

    u_max = 500.0
    u_min = -1.0

    W_x = jnp.array([100.0, 5.0])  # (n_x,) -> (2,) State weights
    W_u = jnp.array([1e-3])        # (n_u,) -> (1,) Control weights
    W_air = 2000.0       # Increased for discrete activation

    N_horizon = 50
    max_iter_mpc = 20

    # FDDP / Merit Function Weights
    mu_defect = 1000.0

cfg = Config()

# ==========================================
# 1. DYNAMICS (Exact Parity)
# ==========================================

def compute_signed_distance(q):
    """
    Computes the signed distance to the contact surface (phi).
    
    Args:
        q: Generalized coordinates (n_q,)
        
    Returns:
        phi: Signed distance (n_c,)
    """
    # For the pogo stick point mass, q is just the height.
    # TODO In a general body, you would do phi_{i,k+1} ~ phi_{i,k} + gradient of forward dynamics w.r.t. q_{i,k} * \delta q_{i,k}. This is also phi_{i,k} + v_{i,k+1} * \delta t, which we can solve by letting phi_{i,k+1} = 0 (signorini condition).
    # For now, just return the height at the current time step.
    
    # Handle scalar case
    if q.ndim == 0:
        return jnp.array([q])
        
    # Handle vector case (taking first element as height)
    return jnp.array([q[0]])

def compute_common_terms(q, q_dot, u):
    """
    Computes terms A and b from Equation 11.
    
    Args:
        q: Configuration scalar or (1,)
        q_dot: Velocity scalar or (1,)
        u: Control scalar or (1,)

    Returns:
        A: Delassus matrix (n_c, n_c) -> (1, 1)
        b: Contact velocity of unconstrained motion (n_c,) -> (1,)
        M_inv: Inverse mass matrix (n_v, n_v) -> (1, 1)
        J: Contact Jacobian (n_c, n_v) -> (1, 1)
        phi: Signed distance to contact (n_c,) or scalar
    """
    # TODO the paper separates J into [J_c; J^n_s] for non separating contact scenarios. Need to reorder J, M, labmda accordingly.

    # --- Dynamics Model ---
    # M: Mass Matrix (Scalar for point mass), TODO Need to calculate/load in when not fully actuated
    M = cfg.m * jnp.eye(1)             # (n_v, n_v) -> (1, 1)
    M_inv = jnp.linalg.inv(M)          # (n_v, n_v) -> (1, 1)
    
    # h: Bias vector (Coriolis + Gravity)
    # Dynamics: M*q_ddot + h = B*u + J^T*lambda
    # h = [m * g]
    h = jnp.array([cfg.m * cfg.g])     # (n_v,) -> (1,)
    
    # B: Input Matrix (TODO Need to calculate/load in when not fully actuated)
    # This could potentially be state-dependent
    phi = compute_signed_distance(q) # (n_c,)
    B = cfg.B_fn(phi)                 # (n_v, n_u) -> (1, 1)
    
    # J: Contact Jacobian (TODO Need to calculate/load in when not fully actuated)
    J = jnp.array([[1.0]])             # (n_c, n_v) -> (1, 1)
    
    # --- Equation 11 Components ---
    # b: Contact velocity of unconstrained motion
    # b = J * M^{-1} * ( (-h + B*u)*dt + M*q_dot )
    
    impulse_non_contact = (-h + B @ jnp.atleast_1d(u)) * cfg.dt # (n_v,) -> (1,)
    momentum_inertial = M @ jnp.atleast_1d(q_dot)               # (n_v,) -> (1,)
    
    # Compute q_dot_unconstrained (velocity if lambda=0)
    # q_dot_unc = M^{-1} * (impulse_non_contact + momentum_inertial)
    q_dot_unc = M_inv @ (impulse_non_contact + momentum_inertial) # (n_v,) -> (1,)
    
    b = J @ q_dot_unc # (n_c,) -> (1,)
    
    # Drift Compensation (Appendix B)
    # Modify b to include the drift term: phi / dt
    # This ensures v_next^n + phi/dt = 0
    
    drift_term = phi / cfg.dt
    b = b + drift_term # (n_c,) -> (1,)
    
    # A: Delassus Matrix
    # A = J * M^{-1} * J^T
    # TODO need to do some calculation for E_s (converts normal portion of lambda to full lambda, section 3.4.3)
    # I think E_s is identity for 1-D case.
    A = J @ M_inv @ J.T # (n_c, n_c) -> (1, 1)
    
    return A, b, M_inv, J, phi

def solve_lcp_pgs(A, b, max_iter=50):
    """
    Solves LCP: 0 <= lambda _|_ A*lambda + b >= 0
    Using Projected Gauss-Seidel.
    Generalized for multi-contact (A is matrix, b is vector).
    """
    n = b.shape[0]
    
    # Optimization for scalar case to avoid scan overhead
    if n == 1:
        val = -b[0] / (A[0, 0] + 1e-8)
        return jnp.array([jnp.maximum(0.0, val)])

    lambda_init = jnp.zeros_like(b)
    
    def iteration(lam, _):
        def row_update(lam_curr, i):
            # Gauss-Seidel step for row i
            # lambda_i = max(0, -(b_i + sum_{j!=i} A_ij lambda_j) / A_ii)
            
            # sigma includes A_ii * lambda_i, so we subtract it back
            sigma = jnp.dot(A[i], lam_curr) - A[i, i] * lam_curr[i]
            val = -(b[i] + sigma) / A[i, i]
            
            # Projection
            val_clamped = jnp.maximum(0.0, val)
            
            # Update entry i in the current lambda vector
            return lam_curr.at[i].set(val_clamped), None
            
        lam_next, _ = jax.lax.scan(row_update, lam, jnp.arange(n))
        return lam_next, None

    lambda_sol, _ = jax.lax.scan(iteration, lambda_init, None, length=max_iter)
    return lambda_sol

@custom_vjp
def solve_contact_impulse(q, q_dot, u):
    """
    Forward Pass: Solves for contact impulse lambda based on Hard Contact Model.
    Uses Signorini Condition (Eq 9a).
    Generalized for multi-contact using PGS solver.
    
    Args:
        q: configuration
        q_dot: velocity
        u: control
    Returns:
        lambda_c: contact impulse (n_c,)
    """
    A, b, M_inv, J, phi = compute_common_terms(q, q_dot, u)
    
    # Solving LCP: 0 <= lambda _|_ (A*lambda + b) >= 0
    # Use Projected Gauss-Seidel for general multi-contact case
    lambda_c = solve_lcp_pgs(A, b)
    
    return lambda_c

def solve_contact_impulse_fwd(q, q_dot, u):
    lambda_c = solve_contact_impulse(q, q_dot, u)
    return lambda_c, (lambda_c, q, q_dot, u)

def solve_contact_impulse_bwd(res, g_lambda):
    """
    Backward Pass: Computes gradients using Relaxed Complementarity Constraints.
    Uses implicit differentiation of Eq 19 via VJP to handle gradients of A and b.
    """
    lambda_c, q, q_dot, u = res
    
    (A, b, M_inv, J, phi), vjp_fn = jax.vjp(compute_common_terms, q, q_dot, u)
    
    # 2. Relaxed Inverse Hessian
    lam_sq = lambda_c**2 + 1e-6
    D_mat = jnp.diag(jnp.atleast_1d(1.0 / lam_sq))
    H_relaxed = A + cfg.rho * D_mat
    
    # 3. Adjoint Vector
    H_inv = jnp.linalg.inv(H_relaxed)
    nu = -jnp.atleast_1d(g_lambda) @ H_inv
    
    # 4. Target Gradients for A and b
    grad_b_target = nu
    grad_A_target = jnp.outer(nu, lambda_c)
    
    # 5. Backpropagate
    grad_M_inv = jnp.zeros_like(M_inv)
    grad_J = jnp.zeros_like(J)
    grad_phi = jnp.zeros_like(phi)
    grad_q, grad_q_dot, grad_u = vjp_fn((grad_A_target, grad_b_target, grad_M_inv, grad_J, grad_phi))
    
    return grad_q, grad_q_dot, grad_u

solve_contact_impulse.defvjp(solve_contact_impulse_fwd, solve_contact_impulse_bwd)

def step_cimpc(state, u):
    """
    Performs one step of the Contact-Implicit Dynamics.
    Maps to Equation 6.
    
    Args:
        state: (n_x,) -> (2,) [q, q_dot]
        u: scalar or (1,)
    Returns:
        next_state: (n_x,) -> (2,)
    """
    q, q_dot = state

    u = jnp.clip(u, cfg.u_min, cfg.u_max)
    
    # 1. Compute Contact Impulse (lambda)
    # This internally solves the optimization (Eq 7/12)
    lambda_c = solve_contact_impulse(q, q_dot, u) # (n_c,)
    
    # 2. Update Generalized Velocity (Eq 6)
    # q_dot_next = M^{-1}( (-h + Bu)dt + M*q_dot + J^T*lambda )
    
    # Re-define terms locally for clarity of Eq 6
    M = cfg.m * jnp.eye(1)    # (n_v, n_v) -> (1, 1)
    M_inv = jnp.linalg.inv(M) # (n_v, n_v) -> (1, 1)
    h = M @ jnp.array([cfg.g]) # Bias vector (n_v,)
    B = cfg.B_fn(q)         # (n_v, n_u)
    J = jnp.array([[1.0]])    # (n_c, n_v)
    
    # Term 1: Non-contact forces integral
    term_forces = (-h + B @ jnp.atleast_1d(u)) * cfg.dt # (n_v,)
    
    # Term 2: Inertial momentum
    term_inertial = M @ jnp.atleast_1d(q_dot) # (n_v,)
    
    # Term 3: Contact Impulse
    term_contact = J.T @ jnp.atleast_1d(lambda_c) # (n_v,)
    
    # Final Velocity Update
    q_dot_next = M_inv @ (term_forces + term_inertial + term_contact) # (n_v,)
    
    # 3. Update Generalized Coordinate (Eq 6 - Semi-implicit)
    # q_next = q (+) q_dot_next * dt
    q_next = q + q_dot_next * cfg.dt # (n_q,)
    
    return jnp.array([jnp.squeeze(q_next), jnp.squeeze(q_dot_next)]) # (n_x,)

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

    # Cost to encourage foot clearance (Air Cost)
    # Only applied when near ground but should be in air (controlled by weight_air)
    is_ground = jax.nn.sigmoid(-50.0 * (state[0] - 0.05))
    c_air = weight_air * is_ground * (state[0]**2)
    
    return c_reg + c_air

def terminal_cost(state, x_ref):
    x_err = state - x_ref
    return jnp.sum((cfg.W_x * 10.0) * x_err**2)

# ==========================================
# 3. FDDP SOLVER (Multiple Shooting Variant)
# ==========================================

def compute_defects(x_traj, u_traj):
    def single_defect(x_curr, u_curr, x_next):
        return step_cimpc(x_curr, u_curr) - x_next
    return jax.vmap(single_defect)(x_traj[:-1], u_traj, x_traj[1:])

def get_derivatives(x_traj, u_traj, x_refs, u_refs, air_weights):
    def step_derivs(x, u, xr, ur, w_air):
        # Dynamics derivatives (A, B)
        fx, fu = jax.jacrev(step_cimpc, (0, 1))(x, u)
        
        # Cost derivatives (l_x, l_u, l_xx, l_uu, l_xu)
        cost_fn = lambda _x, _u: running_cost(_x, _u, xr, ur, w_air)
        lx, lu = jax.grad(cost_fn, (0, 1))(x, u)
        lxx = jax.hessian(cost_fn, 0)(x, u)
        luu = jax.hessian(cost_fn, 1)(x, u)
        lxu = jax.jacfwd(jax.grad(cost_fn, 0), 1)(x, u)
        
        # Reshaping
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
        V_x, V_xx, u_nom = carry
        
        # Multiple Shooting: Incorporate defects into Value function propagation
        # V_x_plus = V_x + V_xx * defect (approximation)
        V_x_plus = V_x + V_xx @ defects[i]

        # Q-function expansion
        Q_x = lx[i] + fx[i].T @ V_x_plus
        Q_u = lu[i] + fu[i].T @ V_x_plus
        Q_xx = lxx[i] + fx[i].T @ V_xx @ fx[i]
        Q_uu = luu[i] + fu[i].T @ V_xx @ fu[i]
        Q_ux = lxu[i].T + fu[i].T @ V_xx @ fx[i]

        # Regularization
        Q_uu_reg = Q_uu + reg_mu * jnp.eye(1)

        # Feedforward (k) and Feedback (K) gains
        k_open = -jnp.linalg.solve(Q_uu_reg, Q_u)
        K_open = -jnp.linalg.solve(Q_uu_reg, Q_ux)

        # Box-DDP: Handling control limits
        u_prospective = u_nom + k_open
        is_clamped = (u_prospective > cfg.u_max) | (u_prospective < cfg.u_min)
        k_clamped = jnp.clip(u_prospective, cfg.u_min, cfg.u_max) - u_nom

        k = jnp.where(is_clamped, k_clamped, k_open)
        K = jnp.where(is_clamped, 0.0 * K_open, K_open)

        # Propagate Value Function
        V_x_prev = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
        V_xx_prev = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
        
        return (V_x_prev, V_xx_prev, u_traj[i-1]), (k, K)

    # Scan backwards
    _, (ks, Ks) = jax.lax.scan(loop_body, (V_x, V_xx, u_traj[-1]), jnp.arange(N)[::-1])
    return ks[::-1], Ks[::-1]

def forward_pass_fddp(x0, x_traj, u_traj, defects, ks, Ks, alpha):
    def scan_fn(carry, inputs):
        x_curr = carry
        i, u_bar, x_bar, k, K, defect = inputs
        
        dx = x_curr - x_bar
        u_new = u_bar + alpha * k + K @ dx
        u_new = jnp.clip(u_new, cfg.u_min, cfg.u_max)
        
        # Rollout with new control
        x_pred = step_cimpc(x_curr, u_new)
        
        # Multiple Shooting: Correct next state by (1-alpha)*defect
        # This ensures that if alpha=0, we recover the original trajectory (with gaps)
        x_next = x_pred - (1.0 - alpha) * defect
        
        return x_next, (x_next, u_new)
    
    inputs = (jnp.arange(u_traj.shape[0]), u_traj, x_traj[:-1], ks, Ks, defects)
    _, (x_new_traj, u_new_traj) = jax.lax.scan(scan_fn, x0, inputs)
    
    return jnp.vstack([x0, x_new_traj]), u_new_traj

def calculate_merit(x_traj, u_traj, defects, x_refs, u_refs, air_weights):
    rc = jax.vmap(running_cost)(x_traj[:-1], u_traj, x_refs[:-1], u_refs, air_weights)
    tc = terminal_cost(x_traj[-1], x_refs[-1])
    cost_total = jnp.sum(rc) + tc
    infeasibility = jnp.sum(jnp.abs(defects))
    return cost_total + cfg.mu_defect * infeasibility, cost_total

# @partial(jax.jit, static_argnums=(6,))
def solve_fddp(x0, x_traj_guess, u_traj_guess, x_refs, u_refs, air_weights, max_iter):
    """
    Main FDDP Loop.
    
    Args:
        x0: Initial state (n_x,)
        x_traj_guess: (N+1, n_x)
        u_traj_guess: (N, n_u)
    Returns:
        x_traj: Optimized state trajectory (N+1, n_x)
        u_traj: Optimized control trajectory (N, n_u)
    """
    x_traj, u_traj = x_traj_guess, u_traj_guess

    for i in range(max_iter):
        defects = compute_defects(x_traj, u_traj)
        merit_current, _ = calculate_merit(x_traj, u_traj, defects, x_refs, u_refs, air_weights)

        derivs = get_derivatives(x_traj, u_traj, x_refs, u_refs, air_weights)
        ks, Ks = backward_pass(derivs, x_traj, u_traj, defects)

        best_merit = merit_current
        best_x, best_u = x_traj, u_traj

        # Line Search
        for alpha in [1.0, 0.8, 0.5, 0.2, 0.05]:
            xn, un = forward_pass_fddp(x0, x_traj, u_traj, defects, ks, Ks, alpha)
            defects_new = compute_defects(xn, un)
            merit_new, _ = calculate_merit(xn, un, defects_new, x_refs, u_refs, air_weights)

            improved = merit_new < best_merit
            
            best_merit = jnp.where(improved, merit_new, best_merit)
            
            # jnp.where works on arrays, so x_traj and u_traj update cleanly
            best_x = jnp.where(improved, xn, best_x)
            best_u = jnp.where(improved, un, best_u)

        x_traj, u_traj = best_x, best_u
    
    return x_traj, u_traj

# ==========================================
# 4. SIMULATION
# ==========================================

def run_mpc_simulation(N_steps):
    N = cfg.N_horizon
    x_current = jnp.array([1.0, 0.0]) # Initial State: Height 0m, Velocity 0
    u_traj = jnp.zeros((N,1))
    
    # Initial guess: just rollout zero control
    def rollout_pure(x, u): return step_cimpc(x, u), step_cimpc(x, u)
    _, x_traj = jax.lax.scan(rollout_pure, x_current, u_traj)
    x_traj = jnp.vstack([x_current, x_traj])

    x_refs = jnp.tile(jnp.array([1.5, 0.0]), (N+1, 1)) # Target: 1.5m
    u_refs = jnp.tile(jnp.array([0.0]), (N, 1))
    
    # Carry state: (x_curr, x_traj, u_traj, time_on_ground_float)
    init_carry = (x_current, x_traj, u_traj, 0.0)
    
    def sim_step(carry, t):
        x_curr, x_tr, u_tr, tog = carry
        
        # Heuristic to encourage flight phases (Air Time Cost)
        # JIT-compatible logic
        is_ground = x_curr[0] < 0.05
        tog_next = jnp.where(is_ground, tog + 1.0, 0.0)
        
        threshold_time = 12.0
        current_air_weight = jnp.where(tog_next > threshold_time, cfg.W_air, 0.0)
        air_weights = jnp.ones((N,)) * current_air_weight
        
        # Solve MPC
        x_tr_opt, u_tr_opt = solve_fddp(x_curr, x_tr, u_tr, x_refs, u_refs, air_weights, cfg.max_iter_mpc)
        
        # Apply Control
        u_applied = u_tr_opt[0]
        x_next = step_cimpc(x_curr, u_applied)
        
        # Compute Cost for logging
        current_cost = total_cost(x_tr_opt, u_tr_opt, x_refs, u_refs, air_weights)
        
        # Warm Start Shift
        u_tr_next = jnp.roll(u_tr_opt, -1, axis=0).at[-1].set(0.0)
        x_tr_next = jnp.roll(x_tr_opt, -1, axis=0)
        
        # Get new terminal guess
        x_term_guess = step_cimpc(x_tr_next[-2], 0.0)
        x_tr_next = x_tr_next.at[-1].set(x_term_guess)
        x_tr_next = x_tr_next.at[0].set(x_next) # Set first state to new current
        
        new_carry = (x_next, x_tr_next, u_tr_next, tog_next)
        # Output: x_next, u_applied, cost
        return new_carry, (x_next, u_applied, current_cost)

    final_carry, (x_hist, u_hist, cost_hist) = jax.lax.scan(sim_step, init_carry, jnp.arange(N_steps))
    
    # Prepend initial state to x_hist to match original structure
    x_hist_full = jnp.vstack([x_current.reshape(1, -1), x_hist])
    
    return x_hist_full, u_hist, cost_hist    

# ==========================================
# 5. PLOTTING
# ==========================================
if __name__ == "__main__":
    run_fast_jit = jax.jit(run_mpc_simulation, static_argnums=(0,))
    
    x_hist, u_hist, cost_hist = run_fast_jit(150)
    
    data = {
        'x': x_hist,
        'u': u_hist,
        'cost': cost_hist
    }
    x_hist = jnp.array(data['x'])
    u_hist = jnp.array(data['u']).flatten()
    t_axis = jnp.arange(len(x_hist)) * cfg.dt

    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.title("Pogo Hopping")
    plt.plot(t_axis, x_hist[:, 0], label="Height (q)")
    plt.axhline(1.5, color='r', linestyle='--', label="Target")
    plt.axhline(0.0, color='k', linewidth=2, label="Ground")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.step(t_axis[:-1], u_hist, where='post', color='orange', label="Control (u)")
    plt.ylabel("Force (N)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(t_axis[:-1], data['cost'], color='purple', label="Cost")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pogo_8.png", dpi=100)
    print("Saved plot to pogo_8.png")




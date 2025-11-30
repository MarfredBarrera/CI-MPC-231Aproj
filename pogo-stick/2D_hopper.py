import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import custom_vjp
import numpy as np
import matplotlib.animation as animation
from matplotlib.patches import Circle

# ==========================================
# 0. CONFIG & DYNAMICS (Your Existing Code)
# ==========================================
class Config:
    n_q, n_v, n_u = 3, 3, 2
    m, I, l = 5.0, 1.0, 1.0
    g, dt, rho = 9.81, 0.01, 1.0 # Smaller dt for smoother physics verification
    u_max = jnp.array([50.0, 1000.0])
    u_min = jnp.array([-50.0, 0.0])

cfg = Config()

# --- Re-pasting the core dynamics functions for self-containment ---
def get_foot_kinematics(q, q_dot):
    x, y, th = q
    vx, vy, w = q_dot
    s, c = jnp.sin(th), jnp.cos(th)
    p_foot = jnp.array([x + cfg.l*s, y - cfg.l*c])
    v_foot = jnp.array([vx + cfg.l*c*w, vy + cfg.l*s*w])
    return p_foot, v_foot

def compute_common_terms(q, q_dot, u):
    x, y, th = q
    M_inv = jnp.diag(jnp.array([1.0/cfg.m, 1.0/cfg.m, 1.0/cfg.I]))
    h = jnp.array([0.0, cfg.m * cfg.g, 0.0])
    
    # B Matrix Logic
    p_foot, _ = get_foot_kinematics(q, q_dot)
    # Mask: 1 if touching ground (approx), 0 if air. 
    # For testing physics, we want checking contact logic to be robust
    contact_mask = jax.nn.sigmoid(-100.0 * (p_foot[1] - 0.05))
    s, c = jnp.sin(th), jnp.cos(th)
    B = jnp.array([[0.0, -s*contact_mask], [0.0, c*contact_mask], [1.0, 0.0]])
    
    # J Matrix Logic
    J = jnp.vstack([
        jnp.array([0.0, 1.0, cfg.l * s]),      # Normal
        jnp.array([1.0, 0.0, cfg.l * c]),      # Tangent +
        jnp.array([-1.0, 0.0, -cfg.l * c])     # Tangent -
    ])
    
    # Bias b (unconstrained vel) + Baumgarte
    impulse_forces = (-h + B @ u) * cfg.dt
    vel_star = q_dot + M_inv @ impulse_forces
    b = J @ vel_star
    b = b.at[0].set(b[0] + p_foot[1] / cfg.dt) # Baumgarte stabilization
    
    A = J @ M_inv @ J.T
    return A, b, M_inv, J

def solve_lcp_pgs(A, b):
    # Simple PGS for sanity check
    n = b.shape[0]
    lam = jnp.zeros_like(b)
    for _ in range(50):
        for i in range(n):
            sigma = jnp.dot(A[i], lam) - A[i,i]*lam[i]
            val = -(b[i] + sigma) / (A[i,i] + 1e-6)
            lam = lam.at[i].set(jnp.maximum(0.0, val))
    return lam

def step_dynamics_test(state, u):
    q, q_dot = state[:3], state[3:]
    u = jnp.clip(u, cfg.u_min, cfg.u_max)
    
    # 1. Solve Contact
    A, b, M_inv, J = compute_common_terms(q, q_dot, u)
    lambda_c = solve_lcp_pgs(A, b)
    
    # 2. Integrate
    # Re-compute B locally for integration
    p_foot, _ = get_foot_kinematics(q, q_dot)
    contact_mask = jax.nn.sigmoid(-100.0 * (p_foot[1] - 0.05))
    s, c = jnp.sin(q[2]), jnp.cos(q[2])
    B = jnp.array([[0.0, -s*contact_mask], [0.0, c*contact_mask], [1.0, 0.0]])
    h = jnp.array([0.0, cfg.m*cfg.g, 0.0])
    
    forces = (-h + B @ u) * cfg.dt
    contact = J.T @ lambda_c
    
    q_dot_next = q_dot + M_inv @ (forces + contact) # (Assuming diagonal M, M*qdot term simplifies)
    q_next = q + q_dot_next * cfg.dt
    
    return jnp.concatenate([q_next, q_dot_next]), lambda_c

def run_moving_right_test():
    # Initial State: 
    # Hip at 1.2m
    # Leg angled BACK slightly (-0.15 rad) so the kick pushes us Right
    # Moving down slightly to trigger contact soon
    x0 = jnp.array([0.0, 1.2, -0.15, 
                    0.0, -1.0, 0.0])
    
    steps = 150 # 3 seconds at 0.02dt (if dt=0.02)
    # Note: Ensure cfg.dt is consistent. Let's force it here for the test.
    dt = 0.01 
    cfg.dt = dt 
    
    state = x0
    x_hist = [x0]
    u_hist = []
    
    print("Simulating 'Hop Right'...")
    
    for _ in range(steps):
        # Extract State
        q, q_dot = state[:3], state[3:]
        x, y, th = q
        
        # --- 1. SENSOR ---
        # Where is the foot?
        p_foot, _ = get_foot_kinematics(q, q_dot)
        foot_height = p_foot[1]
        is_contact = foot_height < 0.02
        
        # --- 2. CONTROLLER ---
        u_test = jnp.array([0.0, 800.0])


        
        # # A. VERTICAL SUPPORT (The "Spring")
        # # Only fire if touching ground. 
        # # Force = Constant high kick to launch us
        # if is_contact:
        #     # Apply max force to jump
        #     u_force = 800.0 
            
        #     # B. ATTITUDE CONTROL (Keep upright)
        #     # If we just kick with angled leg, body will spin. 
        #     # Apply torque to fight the rotation induced by the kick.
        #     # Simple P-controller on Body Angle (which is implicit in theta + geometry)
        #     # Let's just try to keep theta constant for this open loop test.
        #     # u_torque = -kp * (th - target) - kd * dth
        #     u_torque = -200.0 * (th - (-0.15)) - 10.0 * q_dot[2]
            
        #     u_test = jnp.array([u_torque, u_force])
        # else:
        #     # IN AIR: Recovery
        #     # Reset leg angle to be ready for next landing? 
        #     # For this short test, just hold the angle.
        #     u_torque = -100.0 * (th - (-0.15)) - 5.0 * q_dot[2]
        #     u_test = jnp.array([u_torque, 0.0])

        # --- 3. PHYSICS STEP ---
        next_state, _ = step_dynamics_test(state, u_test)
        
        x_hist.append(next_state)
        u_hist.append(u_test)
        state = next_state

    return jnp.array(x_hist), jnp.array(u_hist)

def animator(x_data, u_data, filename="hopper_jump.gif"):
    """
    Creates a GIF of the 2D Rigid-Leg Hopper.
    
    Args:
        x_data: (N, 6) array [x, y, theta, dx, dy, dtheta]
        u_data: (N, 2) array [Torque, Force]
        filename: Output filename
    """
    # Ensure data is numpy (if it came from JAX)
    x_data = np.array(x_data)
    u_data = np.array(u_data)
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Initialize Plot Elements
    # 1. The Ground
    ground_line = ax.axhline(0, color='black', linewidth=2)
    
    # 2. The Body (Hip)
    body_radius = 0.15
    body_circle = Circle((0, 0), body_radius, color='cornflowerblue', zorder=10)
    ax.add_patch(body_circle)
    
    # 3. The Leg
    leg_line, = ax.plot([], [], 'k-', linewidth=4, zorder=5)
    
    # 4. The Foot
    foot_point, = ax.plot([], [], 'ro', markersize=6, zorder=6)
    
    # 5. Trajectory Trail
    trail, = ax.plot([], [], 'b:', linewidth=1, alpha=0.5)
    
    # 6. Text Indicators
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    force_text = ax.text(0.05, 0.90, '', transform=ax.transAxes)

    def get_leg_coords(state):
        x, y, th = state[:3]
        # Same kinematics as dynamics: p_foot = hip + l * [sin(th), -cos(th)]
        # Note: th=0 is vertical down in dynamics, so we use standard trig
        foot_x = x + cfg.l * np.sin(th)
        foot_y = y - cfg.l * np.cos(th)
        return (x, y), (foot_x, foot_y)

    def init():
        body_circle.center = (0, 0)
        leg_line.set_data([], [])
        foot_point.set_data([], [])
        trail.set_data([], [])
        time_text.set_text('')
        return body_circle, leg_line, foot_point, trail, time_text

    def update(frame):
        state = x_data[frame]
        u = u_data[frame]
        
        # Kinematics
        (hx, hy), (fx, fy) = get_leg_coords(state)
        
        # Update Body
        body_circle.center = (hx, hy)
        
        # Update Leg
        leg_line.set_data([hx, fx], [hy, fy])
        
        # Visualizing Force: Change leg color based on Axial Force u[1]
        # Normalized color: 0 force = Black, Max force = Red
        max_force = 1000.0 # From u_max
        force_intensity = np.clip(u[1] / max_force, 0, 1)
        leg_line.set_color((force_intensity, 0, 0)) # RGB
        
        # Update Foot
        foot_point.set_data([fx], [fy])
        
        # Update Trail (History up to current frame)
        trail_len = 50 # Last 50 frames
        start_idx = max(0, frame - trail_len)
        trail.set_data(x_data[start_idx:frame+1, 0], x_data[start_idx:frame+1, 1])
        
        # Tracking Camera (Window width = 4m)
        window_width = 4.0
        ax.set_xlim(hx - window_width/2, hx + window_width/2)
        ax.set_ylim(-0.5, 2.5) # Keep ground in view
        
        # Text
        time_text.set_text(f"Time: {frame * cfg.dt:.2f} s")
        force_text.set_text(f"Force: {u[1]:.1f} N | Torque: {u[0]:.1f} Nm")
        
        return body_circle, leg_line, foot_point, trail, time_text, force_text

    # Create Animation
    # Interval: dt is simulation step. 
    # If dt=0.01, interval=10ms.
    ms_per_frame = int(cfg.dt * 1000)
    ani = animation.FuncAnimation(fig, update, frames=len(x_data)-1, 
                                  init_func=init, blit=False, interval=ms_per_frame)
    
    print(f"Saving animation to {filename}...")
    ani.save(filename, writer='pillow', fps=30)
    print("Done.")

if __name__ == "__main__":
    # Run the new test
    x_data, u_data = run_moving_right_test()
    
    # Time axis
    dt = 0.01
    t_axis = jnp.arange(len(x_data)) * dt


    animator(x_data,u_data)

    
    
    # # Create Plot
    # plt.figure(figsize=(10, 8))
    
    # # 1. Trajectory (Side View)
    # plt.subplot(2, 2, 1)
    # plt.title("2D Trajectory (Side View)")
    # plt.plot(hip_x, hip_y, 'b-o', markevery=10, label="Hip Path")
    # plt.plot(foot_x_list, foot_y_list, 'k--', alpha=0.3, label="Foot Path")
    
    # # Draw ground
    # plt.axhline(0.0, color='r', linewidth=2)
    # plt.xlabel("Horizontal Position (m)")
    # plt.ylabel("Vertical Height (m)")
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal') # Important to see real jump arc
    
    # # 2. X and Y over time
    # plt.subplot(2, 2, 2)
    # plt.title("Position vs Time")
    # plt.plot(t_axis, hip_y, label="Hip Height (Y)")
    # plt.plot(t_axis, hip_x, label="Hip Pos (X)")
    # plt.axhline(0.0, color='r', linestyle='--', label="Ground")
    # plt.legend()
    # plt.grid(True)
    
    # # 3. Inputs
    # plt.subplot(2, 2, 3)
    # plt.title("Control Inputs")
    # # Pad u_data to match x_data length for plotting
    # u_plot = jnp.vstack([u_data, jnp.array([0.0, 0.0])]) 
    # plt.plot(t_axis, u_plot[:, 1], 'r', label="Force (N)")
    # plt.plot(t_axis, u_plot[:, 0], 'g', label="Torque (Nm)")
    # plt.legend()
    # plt.grid(True)

    # # 4. Theta
    # plt.subplot(2, 2, 4)
    # plt.title("Leg Angle (Theta)")
    # plt.plot(t_axis, theta, 'purple', label="Theta")
    # plt.axhline(-0.15, color='k', linestyle=':', label="Target")
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("2D_hopper.png")



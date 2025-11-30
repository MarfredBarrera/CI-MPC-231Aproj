import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
import argparse

# ==========================================
# 0. CONFIG & DYNAMICS 
# ==========================================
class Config:
    n_q, n_v, n_u = 3, 3, 2
    m, I, l = 5.0, 1.0, 1.0
    g, dt, rho = 9.81, 0.01, 1.0 
    u_max = jnp.array([50.0, 1000.0])
    u_min = jnp.array([-50.0, 0.0])

cfg = Config()

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
    
    p_foot, _ = get_foot_kinematics(q, q_dot)
    contact_mask = jax.nn.sigmoid(-100.0 * (p_foot[1] - 0.05))
    s, c = jnp.sin(th), jnp.cos(th)
    
    # B Matrix: Force acts along leg vector
    B = jnp.array([[0.0, -s*contact_mask], [0.0, c*contact_mask], [1.0, 0.0]])
    
    # J Matrix: Contact Constraints
    # CRITICAL FIX: Apply contact_mask to Tangent rows.
    # Without this, friction acts in the air, pinning the foot's X position.
    J = jnp.vstack([
        jnp.array([0.0, 1.0, cfg.l * s]),               # Normal
        jnp.array([1.0, 0.0, cfg.l * c]) * contact_mask, # Tangent + (Friction)
        jnp.array([-1.0, 0.0, -cfg.l * c]) * contact_mask # Tangent - (Friction)
    ])
    
    impulse_forces = (-h + B @ u) * cfg.dt
    vel_star = q_dot + M_inv @ impulse_forces
    b = J @ vel_star
    b = b.at[0].set(b[0] + p_foot[1] / cfg.dt) 
    
    A = J @ M_inv @ J.T
    return A, b, M_inv, J

def solve_lcp_pgs(A, b):
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
    
    A, b, M_inv, J = compute_common_terms(q, q_dot, u)
    lambda_c = solve_lcp_pgs(A, b)
    
    # Re-compute terms for integration
    p_foot, _ = get_foot_kinematics(q, q_dot)
    contact_mask = jax.nn.sigmoid(-100.0 * (p_foot[1] - 0.05))
    s, c = jnp.sin(q[2]), jnp.cos(q[2])
    B = jnp.array([[0.0, -s*contact_mask], [0.0, c*contact_mask], [1.0, 0.0]])
    h = jnp.array([0.0, cfg.m*cfg.g, 0.0])
    
    forces = (-h + B @ u) * cfg.dt
    contact = J.T @ lambda_c
    
    q_dot_next = q_dot + M_inv @ (forces + contact)
    q_next = q + q_dot_next * cfg.dt
    
    return jnp.concatenate([q_next, q_dot_next]), lambda_c

def run_moving_right_test():
    # Initial State:
    x0 = jnp.array([0.0, 1.3, -0.20, 
                    0.0, -1.0, 0.0])
    
    steps = 300 
    dt = 0.01 
    cfg.dt = dt 
    
    state = x0
    x_hist = [x0]
    u_hist = []
    
    print("Simulating 'Hop Right'...")
    
    for _ in range(steps):
        q, q_dot = state[:3], state[3:]
        th = q[2]
        dth = q_dot[2]
        
        # --- SENSOR ---
        p_foot, _ = get_foot_kinematics(q, q_dot)
        is_contact = p_foot[1] < 0.05
        
        # --- CONTROLLER ---
        target_angle = -0.15 # Angle to prepare for landing (leg forward/back)
        
        if is_contact:
            # === STANCE PHASE ===
            # 1. Pogo Force: High to jump
            u_force = 1500.0 
            
            # 2. Attitude: COMPLIANT
            kp = 150.0
            kd = 10.0
            u_torque = -kp * (th - target_angle) - kd * dth
            
            u_test = jnp.array([u_torque, u_force])
        else:
            # === FLIGHT PHASE ===
            # 1. Force: 
            u_force = 0.0
            
            # 2. Attitude: STIFF
            # Move leg to target angle to catch the next step
            kp = 150.0
            kd = 10.0
            u_torque = -kp * (th - target_angle) - kd * dth
            
            u_test = jnp.array([u_torque, u_force])

        # --- PHYSICS STEP ---
        next_state, _ = step_dynamics_test(state, u_test)
        
        x_hist.append(next_state)
        u_hist.append(u_test)
        state = next_state

    return jnp.array(x_hist), jnp.array(u_hist)


def animator(x_data, u_data, filename="2D_hopper.gif"):
    # Ensure data is numpy
    x_data = np.array(x_data)
    u_data = np.array(u_data)
    
    # Create figure with 2 subplots using correct layout setup
    # height_ratios gives the animation (top) more space than the plot (bottom)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

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

if __name__ == "__main__":

    x_data, u_data = run_moving_right_test()
    animator(x_data, u_data)
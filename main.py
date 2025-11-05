import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import casadi as ca
import dynamics

L0 = dynamics.L0
m = dynamics.m
k = dynamics.k
g = dynamics.g
c = dynamics.c


# --- Initial conditions ---
y0 = L0 + 1.0  # Initial height above the floor (m)
v0 = 0.0     # Initial velocity (m/s)

# Time span for the simulation
t_span = (0, 10)  # Start and end time

def simulate_discrete(dt=0.001, t_max=10.0):
    """Discrete simulation with explicit Euler integration."""
    
    # Initialize
    t = 0.0
    y = y0
    v = v0
    
    # Storage
    time_hist = [t]
    pos_hist = [y]
    vel_hist = [v]
    
    while t < t_max:

        u = dynamics.kappa([y, v])

        y, v = np.array([y, v]) + dynamics.f([y, v], u)*dt
        t += dt
        
        # Store history
        time_hist.append(t)
        pos_hist.append(y)
        vel_hist.append(v)
    
    return np.array(time_hist), np.array(pos_hist), np.array(vel_hist)


if __name__ == "__main__":
    time_hist, pos_hist, vel_hist = simulate_discrete()

    # Animate the results
    from animation import animate
    animate(pos_hist.reshape(1, -1), save_path="hopper_bounce.gif", fps=20)



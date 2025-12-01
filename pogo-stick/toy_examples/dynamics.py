import numpy as np

# --- System parameters ---
m = 1.0  # Mass (kg)
k = 100.0  # Spring stiffness (N/m)
L0 = 1.0  # Spring rest length (m)
g = 9.81  # Acceleration due to gravity (m/s^2)
c = 1.5 # Damping coefficient (N·s/m)

def f(state, u):
    """Discrete dynamics update for the hopper system."""
    y, v = state

    # Force calculation
    force_gravity = -m * g
    force_spring = 0.0
    force_damping = 0.0
    force_input = 0.0

    # Spring force is active only when the spring is compressed against the floor
    if y <= L0:
        compression = L0 - y
        force_spring = k * compression
        force_damping = -c * v
        force_input = u

    # Net force and acceleration
    net_force = force_gravity + force_spring + force_damping + force_input
    a = net_force / m

    return np.array([v, a])




def kappa(state):
    """Simple control law to apply an upward force when the spring is compressed."""
    if state[0] <= L0:
        return 50.0  
    return 0.0
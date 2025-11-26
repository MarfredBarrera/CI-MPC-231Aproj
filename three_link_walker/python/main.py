import jax
import jax.numpy as jnp
import numpy as np
from dynamics import ThreeLinkWalkerDynamics
from controller import inputoutput_linearization_control

def main():
    walker = ThreeLinkWalkerDynamics()
    
    # Initial state
    # q = [x, y, q1, q2, q3]
    q = jnp.array([-0.3827, 0.9239, 2.2253, 3.0107, 0.5236])
    dq = jnp.array([0.8653, 0.3584, -1.0957, -2.3078, 2.0323])
    state = jnp.concatenate([q, dq])
    
    print("Initial State:", state)


    # Simulate
    print("Simulating...")
    
    dt = 0.005
    T = 3.0
    steps = int(T / dt)
    
    t_data = [0.0]
    q_data = [state[:5]]
    
    current_state = state
    t = 0.0
    last_impact_time = -1.0  # To avoid immediate repeated impacts
    
    for _ in range(steps):
        # Controller
        u = inputoutput_linearization_control(current_state)
        
        # Dynamics
        ds = walker.continuous_dynamics(current_state, u)
        dq = ds[:5]
        ddq = ds[5:]
        
        # Euler integration
        q = current_state[:5]
        dq = current_state[5:]
        
        dq_next = dq + ddq * dt
        q_next = q + dq_next * dt

        current_state = jnp.concatenate([q_next, dq_next])
        
        # Check for impact (Swing leg hits ground)
        _, _, _, _, pLeg2 = walker.kinematics(q_next)
        
        if walker.detect_impact(q_next) and (t-last_impact_time > 0.5):
            print(f"Impact detected at time {t:.3f} s")
            current_state = walker.impact_dynamics(current_state)
            last_impact_time = t
            
        t += dt
        t_data.append(t)
        q_data.append(current_state[:5])
        # print(f"Time: {t:.3f} s, State: {current_state}")
    
    print("Running animation...")
    from animation import animate_three_link
    animate_three_link(t_data, q_data, filename="three_link_walker_animation.gif")

if __name__ == "__main__":
    main()

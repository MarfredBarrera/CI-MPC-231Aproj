import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dynamics import ThreeLinkWalkerDynamics

class ThreeLinkAnimator:
    def __init__(self, t_data, q_data):
        self.t_data = t_data
        self.q_data = q_data
        self.walker = ThreeLinkWalkerDynamics()
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1, 15)
        self.ax.set_ylim(-1, 5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        
        # Ground line
        self.ax.plot([-1, 20], [0, 0], 'k-', linewidth=2)
        
        # Walker links
        self.line_torso, = self.ax.plot([], [], 'k-', linewidth=2)
        self.line_leg1, = self.ax.plot([], [], 'r-', linewidth=2) # Stance leg (usually)
        self.line_leg2, = self.ax.plot([], [], 'b-', linewidth=2) # Swing leg (usually)
        
        # Joints/CoMs
        self.pt_hip, = self.ax.plot([], [], 'ko', markersize=7, markerfacecolor='g')
        self.pt_torso, = self.ax.plot([], [], 'ko', markersize=7, markerfacecolor='g')
        self.pt_foot1, = self.ax.plot([], [], 'ro', markersize=7, markerfacecolor='g')
        self.pt_foot2, = self.ax.plot([], [], 'bo', markersize=7, markerfacecolor='g')
        
        self.time_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes)

    def init(self):
        self.line_torso.set_data([], [])
        self.line_leg1.set_data([], [])
        self.line_leg2.set_data([], [])
        self.pt_hip.set_data([], [])
        self.pt_torso.set_data([], [])
        self.pt_foot1.set_data([], [])
        self.pt_foot2.set_data([], [])
        self.time_text.set_text('')
        return self.line_torso, self.line_leg1, self.line_leg2, self.pt_hip, self.pt_torso, self.pt_foot1, self.pt_foot2, self.time_text

    def update(self, frame):
        q = self.q_data[frame]
        t = self.t_data[frame]
        
        # Get kinematics
        # q = [x, y, q1, q2, q3]
        # Note: The dynamics class uses JAX arrays, but for plotting we can pass numpy arrays
        # provided the kinematics function doesn't strictly require JAX types (it usually works if using jnp which is compatible)
        # Or we can just implement the simple kinematics here for speed/simplicity if needed.
        # But let's try using the walker class.
        
        # We need to convert to jax array if it's not already, but usually it handles it.
        # Actually, let's just manually compute positions to avoid JAX overhead in animation loop if possible,
        # or just call the function.
        
        x, y, q1, q2, q3 = q
        lL = self.walker.lL
        lT = self.walker.lT
        
        # Torso Tip (Head) - The MATLAB code plots CoM, but let's plot the full link for visualization?
        # MATLAB: l1 = line([x;pT(1)], [y;pT(2)]...) where pT is CoM.
        # Let's stick to MATLAB behavior: Plot line from Hip to Torso CoM.
        
        # Re-implementing kinematics here to ensure we get exactly what we want for plotting
        # (and to avoid JAX compilation overhead inside the loop if not JITed)
        
        # Torso CoM
        pComTorso = np.array([x + lT * np.sin(q3), 
                              y + lT * np.cos(q3)])
        
        # Leg 1 Foot (Stance)
        angle1 = 3 * np.pi / 2 - (q1 + q3)
        pLeg1 = np.array([x - lL * np.cos(angle1),
                          y - lL * np.sin(angle1)])
                          
        # Leg 2 Foot (Swing)
        angle2 = q2 + q3 - np.pi / 2
        pLeg2 = np.array([x + lL * np.cos(angle2),
                          y - lL * np.sin(angle2)])
        
        # Update lines
        self.line_torso.set_data([x, pComTorso[0]], [y, pComTorso[1]])
        self.line_leg1.set_data([x, pLeg1[0]], [y, pLeg1[1]])
        self.line_leg2.set_data([x, pLeg2[0]], [y, pLeg2[1]])
        
        # Update points
        self.pt_hip.set_data([x], [y])
        self.pt_torso.set_data([pComTorso[0]], [pComTorso[1]])
        self.pt_foot1.set_data([pLeg1[0]], [pLeg1[1]])
        self.pt_foot2.set_data([pLeg2[0]], [pLeg2[1]])
        
        self.time_text.set_text(f'Time = {t:.2f}s')
        
        return self.line_torso, self.line_leg1, self.line_leg2, self.pt_hip, self.pt_torso, self.pt_foot1, self.pt_foot2, self.time_text

    def animate(self, filename=None):
        ani = FuncAnimation(self.fig, self.update, frames=len(self.t_data),
                            init_func=self.init, blit=True, interval=1000/30) # 30 fps approx
        if filename:
            ani.save(filename, writer='pillow', fps=30)
            print(f"Animation saved to {filename}")
        else:
            plt.show()

def animate_three_link(t_data, q_data, filename=None):
    animator = ThreeLinkAnimator(t_data, q_data)
    animator.animate(filename)

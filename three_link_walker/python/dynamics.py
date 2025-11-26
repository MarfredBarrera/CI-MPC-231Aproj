import jax
import jax.numpy as jnp

from matlab_gen.f_gen import f_gen
from matlab_gen.g_gen import g_gen
from matlab_gen.dqPlus_gen import dqPlus_gen


class ThreeLinkWalkerDynamics:
    def __init__(self):
        # Parameters
        self.lL = 1.0 # Length of leg
        self.lT = 0.5 # Length of torso
        self.mL = 5.0 # Mass of leg
        self.mT = 10.0 # Mass of torso
        self.mH = 15.0 # Mass of hip
        self.g = 9.81
        self.JL = 0.0  # Inertia of leg (from MATLAB code)
        self.JT = 0.0  # Inertia of torso (from MATLAB code)

    def kinematics(self, q):
        """
        Computes the positions of the CoMs and feet.
        q = [x, y, q1, q2, q3]
        """
        x, y, q1, q2, q3 = q
        lL, lT = self.lL, self.lT

        # Torso CoM
        pComTorso = jnp.array([x + lT * jnp.sin(q3), 
                               y + lT * jnp.cos(q3)])

        # Leg 1 CoM (Stance leg usually)
        # MATLAB: [x - lL/2*cos(270 - (q1 + q3)); y - lL/2*sin(270 - (q1 + q3))]
        # 270 deg = 3*pi/2
        angle1 = 3 * jnp.pi / 2 - (q1 + q3)
        pComLeg1 = jnp.array([x - (lL / 2) * jnp.cos(angle1),
                              y - (lL / 2) * jnp.sin(angle1)])

        # Leg 2 CoM (Swing leg usually)
        # MATLAB: [x + lL/2*cos(q2 + q3 - 90); y - lL/2*sin(q2 + q3 - 90)]
        # 90 deg = pi/2
        angle2 = q2 + q3 - jnp.pi / 2
        pComLeg2 = jnp.array([x + (lL / 2) * jnp.cos(angle2),
                              y - (lL / 2) * jnp.sin(angle2)])

        # Leg 1 Foot
        pLeg1 = jnp.array([x - lL * jnp.cos(angle1),
                           y - lL * jnp.sin(angle1)])

        # Leg 2 Foot
        pLeg2 = jnp.array([x + lL * jnp.cos(angle2),
                           y - lL * jnp.sin(angle2)])
        
        return pComTorso, pComLeg1, pComLeg2, pLeg1, pLeg2

    def continuous_dynamics(self, state, u):
        f = f_gen(state)
        g = g_gen(state)
        dq = f + g @ u

        return dq

    def impact_dynamics(self, state):
        R = jnp.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1]])
        q_minus = state[:5]
        q_plus = R @ q_minus
        dq_Plus = R @ dqPlus_gen(state)
        state_plus = jnp.concatenate([q_plus, dq_Plus])

        return state_plus

    def detect_impact(self, q):
        # Impact when q1 + q3 = pi + pi/8
        q1 = q[2]
        q3 = q[4]
        impact_angle = jnp.pi + (jnp.pi / 8)
        return q1 + q3 - impact_angle > 0.0
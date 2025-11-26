# Three-Link Walker Simulation

This project implements the dynamics and control of a planar three-link bipedal walker. It includes implementations in both MATLAB (original reference) and Python.

## Project Structure

```
three_link_walker/
├── python/                 # Python implementation
│   ├── dynamics.py         # dynamics of continuous \& discrete mapping
│   ├── controller.py       # Feedback linearization controller
│   ├── animation.py        # Visualization using Matplotlib
│   ├── main.py             # Main simulation loop
│   └── matlab_gen/         # Transferred auto-generated matlab symbolic functions to Python functions
├── matlab/                 # Original MATLAB reference code
│   ├── starter_code.m      # Main script
│   └── gen/                # Generated symbolic functions
└── jax/                    # (TODO:JAX structure with proper dynamics computation)
```

## Dynamics Model

The robot is modeled as a planar three-link chain with:
- **Torso**: Link 3
- **Legs**: Link 1 (Stance) and Link 2 (Swing)
- **Actuation**: Torques applied at each leg ($u_1, u_2$).
- **Contact**: Modeled as a hybrid system with continuous stance phases and discrete impact events when the swing leg hits the ground.

The dynamics are derived using the Lagrangian method:
$$ D(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = B u $$

## Control Strategy

The controller implements **Feedback Linearization** (Input-Output Linearization) to enforce virtual constraints.
- **Outputs ($y$)**: Defined to enforce symmetric walking gait patterns (e.g., torso angle and leg phasing).
- **Control Law**:
  $$ u = (L_g L_f y)^{-1} (v - L_f^2 y) $$
  where $v$ is a PD control term to drive $y \to 0$.

## Getting Started

### Running the Simulation

To run the main Python simulation:

```bash
cd python
python main.py
```

## MATLAB vs. Python Port

The `python/matlab_gen/` directory contains Python functions that were automatically converted from the MATLAB symbolic generation output. This leads to spagetti codes, but easier to implement without figuring out all the jacobians.

Planning to implement fully JAX version of dynamics simulation if required.

## References

1.  **Code Source**: ME 292B (Feedback Control of Legged Robots), UC Berkeley.
2.  **Dynamics**: Westervelt, Eric R., et al. *Feedback control of dynamic bipedal robot locomotion*. CRC press, 2018. (pg. 67)

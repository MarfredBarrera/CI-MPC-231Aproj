import jax.numpy as np

def dpSw_gen(in1):
    dq2 = in1[8]
    dq3 = in1[9]
    dx = in1[5]
    dy = in1[6]
    q2 = in1[3]
    q3 = in1[4]
    t2 = np.pi/2.0
    t3 = -t2
    t4 = q2+q3+t3
    t5 = np.cos(t4)
    t6 = np.sin(t4)
    dpSw = np.array([dx-dq2*t6-dq3*t6,dy-dq2*t5-dq3*t5])
    return dpSw
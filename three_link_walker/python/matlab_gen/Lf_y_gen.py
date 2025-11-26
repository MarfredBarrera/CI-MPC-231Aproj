import jax.numpy as np

def Lf_y_gen(in1):
    dq1 = in1[7]
    dq2 = in1[8]
    dq3 = in1[9]
    Lf_y = np.array([dq3,dq1+dq2+dq3*2.0])
    return Lf_y
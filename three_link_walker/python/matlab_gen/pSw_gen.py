import jax.numpy as np

def pSw_gen(in1):
    q2 = in1[3]
    q3 = in1[4]
    x = in1[0]
    y = in1[1]
    t2 = np.pi/2.0
    t3 = -t2
    t4 = q2+q3+t3
    pSw = np.array([x+np.cos(t4),y-np.sin(t4)])
    return pSw
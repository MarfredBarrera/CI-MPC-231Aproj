import jax.numpy as np

def pComTorso_gen(in1):
    q3 = in1[4]
    x = in1[0]
    y = in1[1]
    pComTorso = np.array([x+np.sin(q3)/2.0,y+np.cos(q3)/2.0])
    return pComTorso
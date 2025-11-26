import jax.numpy as np

def pComLeg1_gen(in1):
    q1 = in1[2]
    q3 = in1[4]
    x = in1[0]
    y = in1[1]
    t2 = np.pi*(3.0/2.0)
    t3 = -t2
    t4 = q1+q3+t3
    pComLeg1 = np.array([x-np.cos(t4)/2.0,y+np.sin(t4)/2.0])
    return pComLeg1
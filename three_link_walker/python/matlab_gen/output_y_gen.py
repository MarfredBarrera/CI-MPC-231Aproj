import jax.numpy as np

def output_y_gen(in1):
    q1 = in1[2]
    q2 = in1[3]
    q3 = in1[4]
    output_y = np.array([q3-np.pi/6.0,q1+q2+q3*2.0-np.pi*2.0])
    return output_y
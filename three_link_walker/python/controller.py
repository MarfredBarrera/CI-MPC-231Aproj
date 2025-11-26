import numpy as np
from matlab_gen.LgLf_y_gen import LgLf_y_gen
from matlab_gen.Lf2_y_gen import Lf2_y_gen
from matlab_gen.y_gen import y_gen
from matlab_gen.Lf_y_gen import Lf_y_gen

def phi(x1, x2):
    a = 0.9
    output = x1 + 1/(2-a) * np.sign(x2) * np.abs(x2)**(2-a)
    return output

def psi(x1, x2):
    a = 0.9
    phi_val = phi(x1, x2)
    output = -np.sign(x2)*np.abs(x2)**a - np.sign(phi_val)*np.abs(phi_val)**(a/(2-a))
    return output

def v(y1, y2, dy1, dy2):
    epsilon = 0.1
    output = np.array([
        1/epsilon**2 * psi(y1, epsilon*dy1),
        1/epsilon**2 * psi(y2, epsilon*dy2)
    ])
    return output

def inputoutput_linearization_control(s):
    LgLf = LgLf_y_gen(s)
    Lf2 = Lf2_y_gen(s)
    y = y_gen(s)
    dy = Lf_y_gen(s)
    
    # MATLAB indices are 1-based, Python are 0-based
    input_v = v(y[0], y[1], dy[0], dy[1])

    # ctl = inv(LgLf) * (-Lf2 + input_v)
    ctl = np.linalg.inv(LgLf) @ (-Lf2 + input_v)
    return ctl

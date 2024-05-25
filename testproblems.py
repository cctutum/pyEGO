import numpy as np

def kriging(x):
    if len(x) != 2:
        raise ValueError("Branin's function is for two variables only")
        
    x1, x2 = x
    
    if x1 < 0 or x1 > 1 or x2 < 0 or x2 > 1:
        raise ValueError("Variable outside of range - use x in {0,1}")
    
    X1 = 15 * x1 - 5
    X2 = 15 * x2
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    d = 6
    e = 10
    ff = 1 / (8 * np.pi)
    
    f = a * (X2 - b * X1**2 + c * X1 - d)**2 + e * (1 - ff) * np.cos(X1) + e + 5 * x1
    
    return f
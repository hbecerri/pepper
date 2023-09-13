import numpy as np

def pxpypz_from_ptetaphi(pt, eta, phi):
    pt = abs(pt)
    px = pt*np.cos(phi)
    py = pt*np.sin(phi)
    pz = pt*np.sinh(eta)
    return px, py, pz                     

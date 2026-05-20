# helpers.py
import numpy as np

def event(i, state, M_event):
    return state + M_event[i]

def competition(cm, cM, normprecip):
    return cm + cM * (1 - normprecip)

def competition1(K0, Kh, precip):
    return 1 / (K0 + Kh * precip)

def allee(M, Ms):
    num = M + Ms
    return num / (30 + num)
    # return 1
    #return (0.1 *M + 0.01 *Ms) / (10 + 0.1 *M + 0.01 * Ms)
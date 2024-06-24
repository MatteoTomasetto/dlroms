# Written by: Matteo Tomasetto (Department of Mechanical Engineering, Politecnico di Milano)
# 
# Scientific articles based on this Python package:
#
# [1] Franco et al., Mathematics of Computation (2023).
#     A deep learning approach to reduced order modelling of parameter dependent partial differential equations.
#     DOI: https://doi.org/10.1090/mcom/3781.
#
# [2] Franco et al., Neural Networks (2023).
#     Approximation bounds for convolutional neural networks in operator learning.
#     DOI: https://doi.org/10.1016/j.neunet.2023.01.029
#
# [3] Franco et al., Journal of Scientific Computing (2023). 
#     Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces.
#     DOI: https://doi.org/10.1007/s10915-023-02331-1
#
# [4] Vitullo, Colombo, Franco et al., Finite Elements in Analysis and Design (2024).
#     Nonlinear model order reduction for problems with microstructure using mesh informed neural networks.
#     DOI: https://doi.org/10.1016/j.finel.2023.104068
#
# Please cite the Author if you use this code for your work/research.

import numpy as np
import torch
from math import sqrt
from dolfin import assemble, inner, dx
from dlroms.fespaces import asvector
from dlroms.cores import CPU, GPU
from dlroms.dnns import Clock
from IPython.display import clear_output

def l2_norms(snapshots, space, core = GPU):
    l2_norms = core.zeros(snapshots.shape[0])
    try:
        space = space.sub(0).collapse()
    except:
        space = space
    for i in range(snapshots.shape[0]):
        snapshot = asvector(snapshots[i], space)
        l2_norms[i] = sqrt(assemble(inner(snapshot, snapshot) * dx))
    return l2_norms

def l2_mse(true, pred, space, core = GPU):
    return (l2_norms(true - pred, space, core)).mean()

def l2_mre(true, pred, space, core = GPU):
    return (l2_norms(true - pred, space, core) / l2_norms(true, space, core)).mean()

def l2_vect_mse(true_x, pred_x, true_y, pred_y, space, core = GPU):
    return ((l2_norms(true_x - pred_x, space, core)).pow(2) + (l2_norms(true_y - pred_y, space, core)).pow(2)).sqrt().mean()

def l2_vect_mre(true_x, pred_x, true_y, pred_y, space, core = GPU):
    return ((((l2_norms(true_x - pred_x, space, core)).pow(2) + (l2_norms(true_y - pred_y, space, core)).pow(2)).sqrt()) / (((l2_norms(true_x, space, core)).pow(2) + (l2_norms(true_y, space, core)).pow(2)).sqrt())).mean()

def linf(x):
    return x.abs().max(axis = -1)[0]

def linf_mse(true, pred):
    return (linf(true - pred)).mean()

def linf_mre(true, pred):
    return (linf(true - pred) / linf(true)).mean()

def snapshots(n, sampler, core = GPU, verbose = False, filename = None):
    """Samples a collection of snapshots for a given OCP solver."""
    clock = Clock()
    clock.start()
    mu, y, u = [], [], []
    for seed in range(n):
        if(verbose):
            eta = "N/D" if seed == 0 else Clock.shortparse(clock.elapsed()*(n/float(seed)-1.0))
            print("Generating snapshot n.%d... (ETA: %s)." % (seed + 1, eta))
        mu0, y0, u0 = sampler(seed)
        mu.append(mu0)
        y.append(y0)
        u.append(u0)
        clock.stop()
        if(verbose):
            clear_output(wait = True)
    clock.stop()
    if(verbose):
        clear_output()
        print("Snapshots generated. Elapsed time: %s." % clock.elapsedTime())
    mu, y, u = np.stack(mu), np.stack(y), np.stack(u)
    if(filename is None):
        return core.tensor(mu, y, u)
    else:
        np.savez("%s.npz" % filename.replace(".npz",""), mu = mu, y = y, u = u, time = clock.elapsed())
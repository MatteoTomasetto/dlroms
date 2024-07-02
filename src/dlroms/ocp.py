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
from dolfin import Measure, assemble, inner, grad
from dlroms.fespaces import asvector
from dlroms.cores import CPU, GPU
from dlroms.dnns import Clock
from IPython.display import clear_output

class Norm():
    def __init__(self, mesh, space, measure = None, core = GPU):
        self.space = space
        self.measure = Measure("dx", domain = mesh) if(measure is None) else measure
        self.core = core
    
    def norms(self, snapshots):
        raise RuntimeError("No norms method specified!")

    def mse(self, true, pred):
        return (self.norms(true - pred)).mean()

    def mre(self, true, pred):
        return (self.norms(true - pred) / self.norms(true)).mean()
    
    def mse_vect(self, true_x, pred_x, true_y, pred_y):
        return ((self.norms(true_x - pred_x)).pow(2) + (self.norms(true_y - pred_y)).pow(2)).sqrt().mean()

    def mre_vect(self, true_x, pred_x, true_y, pred_y):
        return ((((self.norms(true_x - pred_x)).pow(2) + (self.norms(true_y - pred_y)).pow(2)).sqrt()) / (((self.norms(true_x)).pow(2) + (self.norms(true_y)).pow(2)).sqrt())).mean()

class L2(Norm):
    def __init__(self, mesh, space, measure = None, core = GPU):
        super(L2, self).__init__(mesh, space, measure, core)
    
    def norms(self, snapshots):
        norms = (self.core).zeros(snapshots.shape[0])
        try:
            space = (self.space).sub(0).collapse()
        except:
            space = (self.space)
        for i in range(snapshots.shape[0]):
            snapshot = asvector(snapshots[i], space)
            norms[i] = sqrt(assemble(inner(snapshot, snapshot) * self.measure))
        return norms

class H10(Norm):
    def __init__(self, mesh, space, measure = None, core = GPU):
        super(H10, self).__init__(mesh, space, measure, core)
    
    def norms(self, snapshots):
        norms = (self.core).zeros(snapshots.shape[0])
        try:
            space = (self.space).sub(0).collapse()
        except:
            space = (self.space)
        for i in range(snapshots.shape[0]):
            snapshot = asvector(snapshots[i], space)
            norms[i] = sqrt(assemble(inner(grad(snapshot), grad(snapshot)) * self.measure))
        return norms

class Linf():   
    def norms(self, snapshots):
        return snapshots.abs().max(axis = -1)[0]

    def mse(self, true, pred):
        return (self.norms(true - pred)).mean()

    def mre(self, true, pred):
        return (self.norms(true - pred) / self.norms(true)).mean()

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

from dlroms.dnns import train # Neural networks handling
from dlroms.roms import num2p, POD, projectup, projectdown # Proper orthogonal decomposition
import matplotlib.pyplot as plt

class OCP():
    def __init__(self, Y, U, MU, ntrain):
        self.Y = Y
        self.U = U
        self.MU = MU
        self.ntrain = ntrain

    def POD(self, which, k, decay = True, color = "black"):
        S = self.Y if(which == "state") else self.U
        pod, eig = POD(S[:self.ntrain], k = k)
        
        if decay:
            plt.plot([i for i in range(1, k + 1)], eig, color = color, marker = 's', markersize = 5, linewidth = 1)
            plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
            plt.scatter(k, eig[k-1], color = color, marker = 's', facecolors = 'none', linestyle = '--', s = 100)
            plt.ylabel("${\sigma_{\mathrm{i}}}^{y}$", fontsize=15) if(which == "state") else plt.ylabel("${\sigma_{\mathrm{i}}}^{u}$", fontsize=15)
            plt.title("Singular values decay");
        
        S_POD = projectdown(pod, S).squeeze(-1)
        S_reconstructed = projectup(pod, S_POD)

        return S_POD, S_reconstructed, pod, eig

    def AE(self, which, encoder, decoder, S = None, training = True, save = True, path = 'autoencoder.pt',  *args, **kwargs):
        if S is None:
            S = self.Y if(which == "state") else self.U

        autoencoder = encoder + decoder
        
        if training:
            autoencoder.He() # NN initialization
            train(autoencoder, S, S, self.ntrain, *args, **kwargs)
        else:
            autoencoder.load_state_dict(torch.load(path))
            autoencoder.eval()
        
        autoencoder.freeze()
        S_AE = encoder(S)
        S_reconstructed = decoder(S_AE)

        if save:
            torch.save(autoencoder.state_dict(), path)

        return S_AE, S_reconstructed

    def PODAE(self, which, k, encoder, decoder, training = True, save = True, path = 'autoencoder.pt',  *args, **kwargs):
        
        S_POD, S_reconstructed_POD, pod, eig = self.POD(which, k, decay = False)
        S_DLROM, S_reconstructed_DLROM = self.AE(which, encoder, decoder, S_POD, training, save, path,  *args, **kwargs)
        S_reconstructed = projectup(pod, decoder(S_DLROM))

        return S_DLROM, S_reconstructed, pod, eig


    def redmap(self, phi, Qred, Ured, minmax = False, training = True, save = True, path = 'phi.pt', *args, **kwargs):

        if minmax:
            Qred_std = (Qred - Qred.min()) / (Qred.max() - Qred.min())
            Ured_std = (Ured - Ured.min()) / (Ured.max() - Ured.min())
            QUred = torch.cat((Qred_std, Ured_std), 1)
        else:
            QUred = torch.cat((Qred, Ured), 1)
        
        if training:
            phi.He() # NN initialization
            train(phi, self.MU, QUred, self.ntrain, *args, **kwargs)
        else:
            phi.load_state_dict(torch.load(path))
            phi.eval()
        
        phi.freeze()
        QUred_hat = phi(self.MU)
        if minmax:
            Qred_hat = QUred_hat[:,:Qred.shape[1]] * (Qred.max() - Qred.min()) + Qred.min()
            Ured_hat = QUred_hat[:,Qred.shape[1]:] * (Ured.max() - Ured.min()) + Ured.min()
        else:
            Qred_hat = QUred_hat[:,:Qred.shape[1]]
            Ured_hat = QUred_hat[:,Qred.shape[1]:]

        if save:
            torch.save(phi.state_dict(), path)

        return Qred_hat, Ured_hat

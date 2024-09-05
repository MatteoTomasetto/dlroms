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
import matplotlib.pyplot as plt
from math import sqrt
from dolfin import Measure, assemble, inner, grad
from dlroms.fespaces import asvector
from dlroms.cores import CPU, GPU
from dlroms.dnns import Clock, train
from dlroms.roms import projectup, projectdown
from dlroms.roms import POD as PODcore
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

class OCP():
    def __init__(self, ntrain):
        self.ntrain = ntrain

    def POD(self, S, k, decay = True, color = "black"):

        pod, eig = PODcore(S[:self.ntrain], k = k)
        
        if decay:
            plt.plot([i for i in range(1, k + 1)], eig, color = color, marker = 's', markersize = 5, linewidth = 1)
            plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
            plt.scatter(k, eig[k-1], color = color, marker = 's', facecolors = 'none', linestyle = '--', s = 100)
            plt.title("Singular values decay");
        
        S_POD = projectdown(pod, S).squeeze(-1)
        S_reconstructed = projectup(pod, S_POD)

        return S_POD, S_reconstructed, pod, eig

    def AE(self, S, encoder, decoder, training = True, save = True, path = 'autoencoder.pt',  *args, **kwargs):

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

    def PODAE(self, S, k, encoder, decoder, training = True, save = True, path = 'autoencoder.pt', decay = True, color = "black", *args, **kwargs):
        
        S_POD, S_reconstructed_POD, pod, eig = self.POD(S, k, decay, color)
        S_DLROM, S_reconstructed_DLROM = self.AE(S_POD, encoder, decoder, training, save, path,  *args, **kwargs)
        S_reconstructed = projectup(pod, decoder(S_DLROM))

        return S_DLROM, S_reconstructed, pod, eig


    def redmap(self, phi, MU, OUT = None, minmax = False, load = False, training = True, save = True, path = 'phi.pt', *args, **kwargs):

        if OUT is None:
            if minmax or training:
                raise RuntimeError("Output data required if minmax or training are True!")
        else:
            lengths = [OUT[i].shape[1] for i in range(len(OUT))]
            intervals = np.insert(np.cumsum(lengths), 0, 0)
            OUT_std = []
            minval = []
            maxval = []

            for i in range(len(OUT)):
                minval.append(OUT[i].min() if minmax else 0)
                maxval.append(OUT[i].max() if minmax else 1)
                OUT_std.append((OUT[i] - minval[i]) / (maxval[i] - minval[i])) 
        
            OUT_std = torch.cat(OUT_std, 1)

            if training:
                phi.He() # NN initialization
                train(phi, MU, OUT_std, self.ntrain, *args, **kwargs)
        
        if load:
            phi.load_state_dict(torch.load(path))
            phi.eval()
               
        phi.freeze()
        OUT_hat = phi(MU)
     
        if save:
            torch.save(phi.state_dict(), path)

        OUT_hat_list = []

        if OUT is not None:
            for i in range(len(OUT)):
                OUT_hat_list.append(OUT_hat[:, intervals[i] : intervals[i+1]] * (maxval[i] - minval[i]) + minval[i])

        return OUT_hat_list
        
def train_nested(phi1, phi2, Y0, U, YU, Y1, ntrain, epochs, optim = torch.optim.LBFGS, lr = 1, lossf = None, error = None, verbose = True, until = None, nvalid = 0, conv = num2p, best = False, cleanup = True, dropout = 0.0):

    optimizer = optim(list(phi1.parameters()) + list(phi2.parameters()), lr = lr)

    ntest = len(Y0)-ntrain
    Y0train, Y0test, Y0valid = Y0[:(ntrain-nvalid)], Y0[-ntest:], Y0[(ntrain-nvalid):ntrain]
    Utrain, Utest, Uvalid = U[:(ntrain-nvalid)], U[-ntest:], U[(ntrain-nvalid):ntrain]
    YUtrain, YUtest, YUvalid = YU[:(ntrain-nvalid)], YU[-ntest:], YU[(ntrain-nvalid):ntrain]
    Y1train, Y1test, Y1valid = Y1[:(ntrain-nvalid)], Y1[-ntest:], Y1[(ntrain-nvalid):ntrain]
    
    if(error == None):
        def error(a, b):
            return lossf(a, b)

    err1 = []
    err2 = []
    clock = Clock()
    clock.start()
    bestv1 = np.inf
    bestv2 = np.inf
    tempcode = int(np.random.rand(1)*1000)
        
    validerr1 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Uvalid, phi1(Y0valid)).item())
    validerr2 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Y1valid, phi2(YUvalid)).item())

    for e in range(epochs):
        
        if(dropout>0.0):
            phi1.unfreeze()
            phi2.unfreeze()
            for layer in phi1:
                if(np.random.rand()<=dropout):
                    layer.freeze() 
            for layer in phi2:
                if(np.random.rand()<=dropout):
                    layer.freeze()  
        
        def closure():
            optimizer.zero_grad()
            loss = lossf(Utrain, phi1(Y0train)) + lossf(Y1train, phi2(torch.cat((Y0train, phi1(Y0train)), 1))) #+ lossf(Y1train, phi2(YUtrain))
            loss.backward()
            return loss
        
        optimizer.step(closure)

        with torch.no_grad():
            if(phi1.l2().isnan().item() or phi2.l2().isnan().item()):
                break
            err1.append([error(Utrain, phi1(Y0train)).item(),
                        error(Utest, phi1(Y0test)).item(),
                        validerr1(),
                       ])
            err2.append([error(Y1train, phi2(YUtrain)).item(),
                        error(Y1test, phi2(YUtest)).item(),
                        validerr2(),
                       ])

            if(verbose):
                if(cleanup):
                        clear_output(wait = True)
                
                print("\t\tTrain%s\tTest" % ("\tValid" if nvalid > 0 else ""))
                print("Epoch "+ str(e+1) + ":\t" + conv(err1[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err1[-1][2]))) + "\t" + conv(err1[-1][1]) + ".")
                print("Epoch "+ str(e+1) + ":\t" + conv(err2[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err2[-1][2]))) + "\t" + conv(err2[-1][1]) + ".")

            if(nvalid > 0 and e > 3):
                if((err1[-1][2] > err1[-2][2]) and (err1[-1][0] < err1[-2][0]) and (err2[-1][2] > err2[-2][2]) and (err2[-1][0] < err2[-2][0])):
                        if((err1[-2][2] > err1[-3][2]) and (err1[-2][0] < err1[-3][0]) and (err2[-2][2] > err2[-3][2]) and (err2[-2][0] < err2[-3][0])):
                                break
            if(until!=None):
                if(err1[-1][0] < until and err2[-1][0] < until):
                        break

            if(best and e > 0):
                if(err1[-1][1] < bestv1):
                    bestv1 = err1[-1][1] + 0.0
                    phi1.save("tempPHI1%d" % tempcode)
                if(err2[-1][1] < bestv2):
                    bestv2 = err2[-1][1] + 0.0
                    phi2.save("tempPHI2%d" % tempcode)
            
    if(best):
        try:
            phi1.load("tempPHI1%d" % tempcode) 
            for file in phi1.files("tempPHI1%d" % tempcode):
                os.remove(file)
            phi2.load("tempPHI2%d" % tempcode) 
            for file in phi2.files("tempPHI2%d" % tempcode):
                os.remove(file)
        except:
            None
    clock.stop()
    if(verbose):
        print("\nTraining complete. Elapsed time: " + clock.elapsedTime() + ".")
    if(dropout>0.0):
        phi1.unfreeze()
        phi2.unfreeze()
    err1 = np.stack(err1)
    err2 = np.stack(err2)
    return err1, err2, clock.elapsed()


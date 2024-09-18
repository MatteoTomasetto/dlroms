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
import os
import matplotlib.pyplot as plt
from math import sqrt
from dolfin import Measure, assemble, inner, grad
from dlroms.fespaces import asvector
from dlroms.cores import CPU, GPU
from dlroms.dnns import Clock
from dlroms.roms import projectup, projectdown, num2p
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
    
    def AE(self, S, encoder, decoder, training = True, save = True, path = 'autoencoder.pt', *args, **kwargs):

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
    
    def latent_policy(self, Y, U, MU, encoder_Y, decoder_Y, encoder_U, decoder_U, policy, training = True, save = True, path = 'NN/', *args, **kwargs):
        
        autoencoder_Y = encoder_Y + decoder_Y
        autoencoder_U = encoder_U + decoder_U

        if training:
            autoencoder_Y.He()
            autoencoder_U.He()
            policy.He()
            train_latent_policy(encoder_Y, decoder_Y, encoder_U, decoder_U, policy, Y, U, MU, self.ntrain, *args, **kwargs)
        else:
            autoencoder_Y.load_state_dict(torch.load(path + 'autoencoder_Y'))
            autoencoder_Y.eval()
            autoencoder_U.load_state_dict(torch.load(path + 'autoencoder_U'))
            autoencoder_U.eval()
            policy.load_state_dict(torch.load(path + 'policy'))
            policy.eval()
        
        autoencoder_Y.freeze()
        autoencoder_U.freeze()
        policy.freeze()

        Y_AE = encoder_Y(Y)
        Y_reconstructed = decoder_Y(Y_AE)
        U_AE = encoder_U(U)
        U_reconstructed = decoder_U(U_AE)
        U_DLROM_hat = policy(torch.cat((MU, Y_AE),1))
        U_hat = decoder_U(U_DLROM_hat)
        if save:
            torch.save(autoencoder_Y.state_dict(), path + 'autoencoder_Y')
            torch.save(autoencoder_U.state_dict(), path + 'autoencoder_U')
            torch.save(policy.state_dict(), path + 'policy')

        return Y_AE, Y_reconstructed, U_AE, U_reconstructed, U_DLROM_hat, U_hat

def train(dnn, input, output, ntrain, epochs, optim = torch.optim.LBFGS, lr = 1, loss = None, error = None, nvalid = 0, verbose = True, notation = '%', batchsize = None, slope = 1.0, until = None, best = False, refresh = True, dropout = 0.0):
    
    conv = (lambda x: num2p(x)) if notation == '%' else (lambda z: ("%.2"+notation) % z)
    optimizer = optim(dnn.parameters(), lr = lr)
    ntest = len(input)-ntrain
    inputtrain, outputtrain, inputtest, outputtest = input[:(ntrain-nvalid)], output[:(ntrain-nvalid)], input[-ntest:], output[-ntest:]
    inputvalid, outputvalid = input[(ntrain-nvalid):ntrain], output[(ntrain-nvalid):ntrain]
    
    if(error == None):
        def error(a, b):
            return loss(a, b)

    err = []
    clock = Clock()
    clock.start()
    bestv = np.inf
    tempcode = int(np.random.rand(1)*1000)
        
    validerr = (lambda : np.nan) if nvalid == 0 else (lambda : error(outputvalid, dnn(inputvalid)).item())

    for e in range(epochs):
        
        if(dropout>0.0):
            dnn.unfreeze()
            for layer in dnn:
                if(np.random.rand()<=dropout):
                    layer.freeze()      
        
        if(batchsize == None):
            def closure():
                optimizer.zero_grad()
                lossf = slope*loss(outputtrain, dnn(inputtrain))
                lossf.backward()
                return lossf
            optimizer.step(closure)
        else:
            indexes = np.random.permutation(ntrain-nvalid)
            nbatch = ntrain//batchsize
            for j in range(nbatch):
                inputbatch = inputtrain[indexes[(j*batchsize):((j+1)*batchsize)]]
                outputbatch = outputtrain[indexes[(j*batchsize):((j+1)*batchsize)]]
                def closure():
                    optimizer.zero_grad()
                    lossf = loss(outputbatch, dnn(inputbatch))
                    lossf.backward()
                    return lossf
                optimizer.step(closure)

        with torch.no_grad():
            if(dnn.l2().isnan().item()):
                break
            err.append([error(outputtrain, dnn(inputtrain)).item(),
                        error(outputtest, dnn(inputtest)).item() if ntest > 0 else np.nan,
                        validerr(),
                    ])
            if(verbose):
                if(refresh):
                        clear_output(wait = True)
                
                print("\t\tTrain%s\tTest" % ("\tValid" if nvalid > 0 else ""))
                print("Epoch "+ str(e+1) + ":\t" + conv(err[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err[-1][2]))) + "\t" + conv(err[-1][1]) + ".")
            if(nvalid > 0 and e > 3):
                if((err[-1][2] > err[-2][2]) and (err[-1][0] < err[-2][0])):
                        if((err[-2][2] > err[-3][2]) and (err[-2][0] < err[-3][0])):
                                break
            if(until!=None):
                if(err[-1][0] < until):
                        break
            if(best and e > 0):
                if(err[-1][1] < bestv):
                        bestv = err[-1][1] + 0.0
                        dnn.save("temp%d" % tempcode)
    
    if(best):
        try:
            dnn.load("temp%d" % tempcode) 
            for file in dnn.files("temp%d" % tempcode):
                os.remove(file)
        except:
            None
    clock.stop()
    if(verbose):
        print("\nTraining complete. Elapsed time: " + clock.elapsedTime() + ".")
    if(dropout>0.0):
        dnn.unfreeze()
    err = np.stack(err)
    return err, clock.elapsed()
        
def train_latent_policy(encoder_Y, decoder_Y, encoder_U, decoder_U, policy, Y, U, MU, ntrain, epochs, batchsize = None, optim = torch.optim.LBFGS, lr = 1, loss = None, error = None, verbose = True, nvalid = 0, notation = '%', best = False, refresh = True):

    conv = (lambda x: num2p(x)) if notation == '%' else (lambda z: ("%.2"+notation) % z)

    autoencoder_Y = encoder_Y + decoder_Y
    autoencoder_U = encoder_U + decoder_U

    optimizer = optim(list(autoencoder_Y.parameters()) + list(autoencoder_U.parameters()) + list(policy.parameters()), lr = lr)

    ntest = len(Y)-ntrain
    Ytrain, Ytest, Yvalid = Y[:(ntrain-nvalid)], Y[-ntest:], Y[(ntrain-nvalid):ntrain]
    Utrain, Utest, Uvalid = U[:(ntrain-nvalid)], U[-ntest:], U[(ntrain-nvalid):ntrain]
    MUtrain, MUtest, MUvalid = MU[:(ntrain-nvalid)], MU[-ntest:], MU[(ntrain-nvalid):ntrain]
    
    if(error == None):
        def error(a, b):
            return loss(a, b)

    err1 = []
    err2 = []
    err3 = []
    err4 = []
    bestv1 = np.inf
    bestv2 = np.inf
    bestv3 = np.inf
    bestv4 = np.inf
    tempcode = int(np.random.rand(1)*1000)
    clock = Clock()
    clock.start()

    validerr1 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Yvalid, autoencoder_Y(Yvalid)).item())
    validerr2 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Uvalid, autoencoder_U(Uvalid)).item())
    validerr3 = (lambda : np.nan) if nvalid == 0 else (lambda : error(torch.cat(encoder_U(Uvalid), policy(torch.cat((MUvalid, encoder_Y(Ytrain)), 1))).item()))
    validerr4 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Uvalid, decoder_U(policy(torch.cat((MUvalid, encoder_Y(Ytrain)), 1)))).item())

    for e in range(epochs):
        
        if(batchsize == None):
            def closure():
                optimizer.zero_grad()
                lossf = loss(Ytrain, autoencoder_Y(Ytrain)) + \
                loss(Utrain, autoencoder_U(Utrain)) + \
                loss(encoder_U(Utrain), policy(torch.cat((MUtrain, encoder_Y(Ytrain)), 1))) + \
                loss(Utrain, decoder_U(policy(torch.cat((MUtrain, encoder_Y(Ytrain)), 1))))
                lossf.backward()
                return lossf
            optimizer.step(closure)
        else:
            indexes = np.random.permutation(ntrain-nvalid)
            nbatch = ntrain//batchsize
            for j in range(nbatch):
                Ybatch = Ytrain[indexes[(j*batchsize):((j+1)*batchsize)]]
                Ubatch = Utrain[indexes[(j*batchsize):((j+1)*batchsize)]]
                MUbatch = MUtrain[indexes[(j*batchsize):((j+1)*batchsize)]]
                def closure():
                    optimizer.zero_grad()
                    lossf = loss(Ybatch, autoencoder_Y(Ybatch)) + \
                    loss(Ubatch, autoencoder_U(Ubatch)) + \
                    loss(encoder_U(Ubatch), policy(torch.cat((MUbatch, encoder_Y(Ybatch)), 1))) + \
                    loss(Ubatch, decoder_U(policy(torch.cat((MUbatch, encoder_Y(Ybatch)), 1))))
                    lossf.backward()
                    return lossf
                optimizer.step(closure)  
                
        with torch.no_grad():
            if(autoencoder_Y.l2().isnan().item() or autoencoder_U.l2().isnan().item() or policy.l2().isnan().item()):
                break
            err1.append([error(Ytrain, autoencoder_Y(Ytrain)).item(),
                        error(Ytest, autoencoder_Y(Ytest)).item(),
                        validerr1(),
                    ])
            err2.append([error(Utrain, autoencoder_U(Utrain)).item(),
                        error(Utest, autoencoder_U(Utest)).item(),
                        validerr2(),
                    ])
            err3.append([error(encoder_U(Utrain), policy(torch.cat((MUtrain, encoder_Y(Ytrain)), 1))).item(),
                        error(encoder_U(Utest), policy(torch.cat((MUtest, encoder_Y(Ytest)), 1))).item(),
                        validerr3(),
                    ])
            err4.append([error(Utrain, decoder_U(policy(torch.cat((MUtrain, encoder_Y(Ytrain)), 1)))).item(),
                        error(Utest, decoder_U(policy(torch.cat((MUtest, encoder_Y(Ytest)), 1)))).item(),
                        validerr4(),
                    ])
        
            if(verbose):
                if(refresh):
                    clear_output(wait = True)

                print("Epoch " + str(e+1))
                print("\t\tTrain%s\tTest" % ("\tValid" if nvalid > 0 else ""))
                print("AE_y \t\t" + conv(err1[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err1[-1][2]))) + "\t" + conv(err1[-1][1]) + ".")
                print("AE_U \t\t" + conv(err2[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err2[-1][2]))) + "\t" + conv(err2[-1][1]) + ".")
                print("Policy_N \t" + conv(err3[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err3[-1][2]))) + "\t" + conv(err3[-1][1]) + ".")
                print("Policy_POD \t" + conv(err4[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err4[-1][2]))) + "\t" + conv(err4[-1][1]) + ".")

            if(best and e > 0):
                if(err1[-1][1] < bestv1 and err2[-1][1] < bestv2 and err3[-1][1] < bestv3 and err4[-1][1] < bestv4):
                    bestv1 = err1[-1][1] + 0.0
                    autoencoder_Y.save("temp_autoencoder_Y_%d" % tempcode)
                    bestv2 = err2[-1][1] + 0.0
                    autoencoder_U.save("temp_autoencoder_U_%d" % tempcode)
                    bestv3 = err3[-1][1] + 0.0
                    bestv4 = err4[-1][1] + 0.0
                    policy.save("temp_policy_%d" % tempcode)
            
    if(best):
        try:
            autoencoder_Y.load("temp_autoencoder_Y_%d" % tempcode) 
            for file in autoencoder_Y.files("temp_autoencoder_Y_%d" % tempcode):
                os.remove(file)
            autoencoder_U.load("temp_autoencoder_U_%d" % tempcode) 
            for file in autoencoder_U.files("temp_autoencoder_U_%d" % tempcode):
                os.remove(file)
            policy.load("temp_policy_%d" % tempcode) 
            for file in policy.files("temp_policy_%d" % tempcode):
                os.remove(file)
        except:
            None
    clock.stop()
    if(verbose):
        print("\nTraining complete. Elapsed time: " + clock.elapsedTime() + ".")

    err1 = np.stack(err1)
    err2 = np.stack(err2)
    err3 = np.stack(err3)
    err4 = np.stack(err4)
    return err1, err2, err3, err4, clock.elapsed()
    """
    def train_latent_loop(encoder_Y, decoder_Y, encoder_U, decoder_U, policy, phi, Y, U, MU, ntrain, epochs, batchsize = None, optim = torch.optim.LBFGS, lr = 1, loss = None, error = None, verbose = True, nvalid = 0, notation = '%', best = False, refresh = True):

        conv = (lambda x: num2p(x)) if notation == '%' else (lambda z: ("%.2"+notation) % z)

        autoencoder_Y = encoder_Y + decoder_Y
        autoencoder_U = encoder_U + decoder_U

        optimizer = optim(list(autoencoder_Y.parameters()) + list(autoencoder_U.parameters()) + list(policy.parameters()) + list(phi.parameters()), lr = lr)

        ntimesteps = U.shape[1]
        nstate = Y.shape[2]
        ncontrol = U.shape[2]
        nparams = MU.shape[2]
        ntest = len(Y)-ntrain
        Ytrain, Ytest, Yvalid = Y[:(ntrain-nvalid)].reshape(-1, nstate), Y[-ntest:].reshape(-1, nstate), Y[(ntrain-nvalid):ntrain].reshape(-1, nstate)
        Y0train, Y0test, Y0valid = Y[:(ntrain-nvalid),:ntimesteps].reshape(-1, nstate), Y[-ntest:,:ntimesteps].reshape(-1, nstate), Y[(ntrain-nvalid):ntrain,:ntimesteps].reshape(-1, nstate)
        Y1train, Y1test, Y1valid = Y[:(ntrain-nvalid),1:].reshape(-1, nstate), Y[-ntest:,1:].reshape(-1, nstate), Y[(ntrain-nvalid):ntrain,1:].reshape(-1, nstate)
        Utrain, Utest, Uvalid = U[:(ntrain-nvalid)].reshape(-1, ncontrol), U[-ntest:].reshape(-1, ncontrol), U[(ntrain-nvalid):ntrain].reshape(-1, ncontrol)
        MUtrain, MUtest, MUvalid = MU[:(ntrain-nvalid)].reshape(-1, nparams), MU[-ntest:].reshape(-1, nparams), MU[(ntrain-nvalid):ntrain].reshape(-1, nparams)

        if(error == None):
            def error(a, b):
                return loss(a, b)

        err1 = []
        err2 = []
        err3 = []
        err4 = []
        err5 = []
        err6 = []
        bestv1 = np.inf
        bestv2 = np.inf
        bestv3 = np.inf
        bestv4 = np.inf
        bestv5 = np.inf
        bestv6 = np.inf
        tempcode = int(np.random.rand(1)*1000)
        clock = Clock()
        clock.start()

        validerr1 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Yvalid, autoencoder_Y(Yvalid)).item())
        validerr2 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Uvalid, autoencoder_U(Uvalid)).item())
        validerr3 = (lambda : np.nan) if nvalid == 0 else (lambda : error(torch.cat(encoder_U(Uvalid), policy(torch.cat((MUvalid, encoder_Y(Y0valid)), 1))).item()))
        validerr4 = (lambda : np.nan) if nvalid == 0 else (lambda : error(Uvalid, decoder_U(policy(torch.cat((MUvalid, encoder_Y(Y0valid)), 1)))).item())
        validerr5 = (lambda : np.nan) if nvalid == 0 else (lambda : error(encoder_Y(Y1valid), phi(torch.cat((MUvalid, encoder_Y(Y0valid), policy(torch.cat((MUvalid, encoder_Y(Y0valid)),1))),1))).item())
        validerr6 = (lambda : np.nan) if nvalid == 0 else (lambda : error(encoder_Y(Y1valid), phi(torch.cat((MUvalid, encoder_Y(Y0valid), encoder_U(Uvalid)), 1))).item())

        for e in range(epochs):
            
            if(batchsize == None):
                def closure():
                    optimizer.zero_grad()
                    lossf = loss(Ytrain, autoencoder_Y(Ytrain)) + \
                    loss(Utrain, autoencoder_U(Utrain)) + \
                    loss(encoder_U(Utrain), policy(torch.cat((MUtrain, encoder_Y(Y0train)), 1))) + \
                    loss(Utrain, decoder_U(policy(torch.cat((MUtrain, encoder_Y(Y0train)), 1)))) + \
                    loss(encoder_Y(Y1train), phi(torch.cat((MUtrain, encoder_Y(Y0train), policy(torch.cat((MUtrain, encoder_Y(Y0train)),1))),1))) + \
                    loss(encoder_Y(Y1train), phi(torch.cat((MUtrain, encoder_Y(Y0train), encoder_U(Utrain)), 1)))
                    lossf.backward()
                    return lossf
                optimizer.step(closure)
            else:
                indexes = np.random.permutation(ntrain-nvalid)
                nbatch = ntrain//batchsize
                for j in range(nbatch):
                    Ybatch = Y[indexes[(j*batchsize):((j+1)*batchsize)]].reshape(-1, nstate)
                    Y0batch = Y[indexes[(j*batchsize):((j+1)*batchsize)],:ntimesteps].reshape(-1, nstate)
                    Y1batch = Y[indexes[(j*batchsize):((j+1)*batchsize)],1:].reshape(-1, nstate)
                    Ubatch = U[indexes[(j*batchsize):((j+1)*batchsize)]].reshape(-1, ncontrol)
                    MUbatch = MU[indexes[(j*batchsize):((j+1)*batchsize)]].reshape(-1, nparams)
                    def closure():
                        optimizer.zero_grad()
                        lossf = loss(Ybatch, autoencoder_Y(Ybatch)) + \
                        loss(Ubatch, autoencoder_U(Ubatch)) + \
                        loss(encoder_U(Ubatch), policy(torch.cat((MUbatch, encoder_Y(Y0batch)), 1))) + \
                        loss(Ubatch, decoder_U(policy(torch.cat((MUbatch, encoder_Y(Y0batch)), 1)))) + \
                        loss(encoder_Y(Y1batch), phi(torch.cat((MUbatch, encoder_Y(Y0batch), policy(torch.cat((MUbatch, encoder_Y(Y0batch)),1))),1))) + \
                        loss(encoder_Y(Y1batch), phi(torch.cat((MUbatch, encoder_Y(Y0batch), encoder_U(Ubatch)), 1)))
                        lossf.backward()
                        return lossf
                    optimizer.step(closure)  
                    
            with torch.no_grad():
                if(autoencoder_Y.l2().isnan().item() or autoencoder_U.l2().isnan().item() or policy.l2().isnan().item() or  phi.l2().isnan().item()):
                    break
                err1.append([error(Ytrain, autoencoder_Y(Ytrain)).item(),
                            error(Ytest, autoencoder_Y(Ytest)).item(),
                            validerr1(),
                        ])
                err2.append([error(Utrain, autoencoder_U(Utrain)).item(),
                            error(Utest, autoencoder_U(Utest)).item(),
                            validerr2(),
                        ])
                err3.append([error(encoder_U(Utrain), policy(torch.cat((MUtrain, encoder_Y(Y0train)), 1))).item(),
                            error(encoder_U(Utest), policy(torch.cat((MUtest, encoder_Y(Y0test)), 1))).item(),
                            validerr3(),
                        ])
                err4.append([error(Utrain, decoder_U(policy(torch.cat((MUtrain, encoder_Y(Y0train)), 1)))).item(),
                            error(Utest, decoder_U(policy(torch.cat((MUtest, encoder_Y(Y0test)), 1)))).item(),
                            validerr4(),
                        ])
                err5.append([error(encoder_Y(Y1train), phi(torch.cat((MUtrain, encoder_Y(Y0train), policy(torch.cat((MUtrain, encoder_Y(Y0train)),1))),1))).item(),
                            error(encoder_Y(Y1test), phi(torch.cat((MUtest, encoder_Y(Y0test), policy(torch.cat((MUtest, encoder_Y(Y0test)),1))),1))).item(),
                            validerr5(),
                        ])
                err6.append([error(encoder_Y(Y1train), phi(torch.cat((MUtrain, encoder_Y(Y0train), encoder_U(Utrain)), 1))).item(),
                            error(encoder_Y(Y1test), phi(torch.cat((MUtest, encoder_Y(Y0test), encoder_U(Utest)), 1))).item(),
                            validerr6(),
                        ])
            
                if(verbose):
                    if(refresh):
                        clear_output(wait = True)

                    print("Epoch " + str(e+1))
                    print("\t\tTrain%s\tTest" % ("\tValid" if nvalid > 0 else ""))
                    print("AE_y \t\t" + conv(err1[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err1[-1][2]))) + "\t" + conv(err1[-1][1]) + ".")
                    print("AE_U \t\t" + conv(err2[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err2[-1][2]))) + "\t" + conv(err2[-1][1]) + ".")
                    print("Policy \t" + conv(err3[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err3[-1][2]))) + "\t" + conv(err3[-1][1]) + ".")
                    print("Policy(POD) \t" + conv(err4[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err4[-1][2]))) + "\t" + conv(err4[-1][1]) + ".")
                    print("Phi \t" + conv(err5[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err5[-1][2]))) + "\t" + conv(err5[-1][1]) + ".")
                    print("Phi(Policy) \t" + conv(err6[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err6[-1][2]))) + "\t" + conv(err6[-1][1]) + ".")

                if(best and e > 0):
                    if(err1[-1][1] < bestv1 and err2[-1][1] < bestv2 and err3[-1][1] < bestv3 and err4[-1][1] < bestv4 and err5[-1][1] < bestv5 and err6[-1][1] < bestv6):
                        bestv1 = err1[-1][1] + 0.0
                        autoencoder_Y.save("temp_autoencoder_Y_%d" % tempcode)
                        bestv2 = err2[-1][1] + 0.0
                        autoencoder_U.save("temp_autoencoder_U_%d" % tempcode)
                        bestv3 = err3[-1][1] + 0.0
                        bestv4 = err4[-1][1] + 0.0
                        policy.save("temp_policy_%d" % tempcode)
                        bestv5 = err5[-1][1] + 0.0
                        bestv6 = err6[-1][1] + 0.0
                        phi.save("temp_phi_%d" % tempcode)
                
        if(best):
            try:
                autoencoder_Y.load("temp_autoencoder_Y_%d" % tempcode) 
                for file in autoencoder_Y.files("temp_autoencoder_Y_%d" % tempcode):
                    os.remove(file)
                autoencoder_U.load("temp_autoencoder_U_%d" % tempcode) 
                for file in autoencoder_U.files("temp_autoencoder_U_%d" % tempcode):
                    os.remove(file)
                policy.load("temp_policy_%d" % tempcode) 
                for file in policy.files("temp_policy_%d" % tempcode):
                    os.remove(file)
                phi.load("temp_phi_%d" % tempcode) 
                for file in phi.files("temp_phi_%d" % tempcode):
                    os.remove(file)
            except:
                None
        clock.stop()
        if(verbose):
            print("\nTraining complete. Elapsed time: " + clock.elapsedTime() + ".")

        err1 = np.stack(err1)
        err2 = np.stack(err2)
        err3 = np.stack(err3)
        err4 = np.stack(err4)
        err4 = np.stack(err5)
        err4 = np.stack(err6)
        return err1, err2, err3, err4, err5, err6, clock.elapsed() """
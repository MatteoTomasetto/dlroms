# Written by: Nicola Rares Franco, Ph.D. (MOX, Department of Mathematics, Politecnico di Milano)
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

from dlroms import fespaces
import dolfin
import numpy as np
import scipy.sparse.linalg as spla
from dlroms.minns import Navigator

class GaussianRandomField(object):
    """Class for managing isotropic Gaussian random fields over general domains. Objects of this class have the following attributes,
    
    Attributes
        n               (int)               Rank at which the field is approximated.
        singvalues      (numpy.ndarray)     Square roots of the covariance kernel eigenvalues.
        eigenfunctions  (numpy.ndarray)     Eigenfunctions of the covariance kernel. This are stored in a k x n matrix, where k is the spatial
                                            dimension (i.e. number of mesh vertices in the discretized domain), while n is the number of computed
                                            eigenfunctions (which equals self.n).                                
    """
    
    def __init__(self, mesh, kernel, upto, domain = None, geodesic_accuracy = None):
        """Constructs a Gaussian random field object.
        
        Input
            mesh                (dolfin.cpp.mesh.Mesh)      Mesh discretizing the spatial domain.
            kernel              (function)                  A function describing the covariance kernel. Such function should only accept a single argument, the
                                                            latter being the distance between two points. Namely, if G is the random field, then cov(|x_i - x_j|)
                                                            should return the covariance between G(x_i) and G(x_j). Clearly, this only allows for isotropic fields.
            upto                (int)                       Number of eigenfunctions to compute. Equivalently, rank at which the process is 
                                                            approximated via its Karhunen-Loeve expansion.
            Optional:
            domain              (mshr.cpp.Geometry)         Domain of reference (optional). Only used for kernels described in terms of geodesic distances.
            geodesic_accuracy   (int)                       Accuracy of the geodesic distance (optional). Only used for kernels described in terms of geodesic
                                                            distances.
        """
        self.cov = kernel
        self.n = upto
        if(domain is None and geodesic_accuracy is None):
            distances = None
        else:
            space = fespaces.space(mesh, 'CG', 1)
            try:
                navigator = Navigator(domain, fespaces.mesh(domain, resolution = geodesic_accuracy))
            except (KeyError, TypeError):
                navigator = Navigator(domain, fespaces.mesh(domain, stepsize = geodesic_accuracy))
            E1 = navigator.finde(fespaces.coordinates(space)).reshape(-1,1)
            E2 = navigator.finde(fespaces.coordinates(space)).reshape(1,-1)
            distances = navigator.D[E1, E2]   
            d = navigator.A.shape[-1]
            extras = np.linalg.norm(((navigator.A+navigator.B+navigator.C)/3.0)[E1].reshape(-1, d) - fespaces.coordinates(space), axis = -1)
            distances = distances + extras.reshape(-1,1) + extras.reshape(1,-1)
        self.svalues, self.eigenfunctions = KarhunenLoeve(mesh, self.cov, self.n, distances = distances)
        self.svalues = np.sqrt(self.svalues) 
        
    def sample(self, seed, coeff = False, upto = None):
        np.random.seed(seed)
        till = upto if upto!= None else sum(1-np.isnan(self.svalues))
        c = np.random.randn(self.n)
        v = np.dot(self.eigenfunctions[:,:till], self.svalues[:till]*c[:till])
        if(coeff):
            return v, c
        else:
            return v      
    
        
def KarhunenLoeve(mesh, covariance, nphis, distances = None):
    """Solves the eigenvalue problem for a given covariance operator.
    
    Input
        mesh        (dolfin.cpp.mesh.Mesh)      Mesh discretizing the spatial domain.
        covariance  (function)                  Covariance kernel (isotropic case). See dlroms.gp.GaussianRandomFields.
        nphis       (int)                       Number of eigenfunctions to compute.

        Optional:
        distances   (numpy.ndarray)             Pairwise distances between x_i and x_j. If None, Euclidean distances are used.
        
    Output
        (tuple of numpy.ndarray). Returns eigenvalues and the eigenfunctions. The former are stored in a vector of length nphis,
        while the latter are written in a matrix k x nphis, where k = mesh.num_vertices().
        
    Remark. The eigenvalue problem is: find lambda and u such that the integral of covariance(|x-y|)u(y)dy equals lambda*u(x).
    The latter is solved via Galerkin projection over the space of P1-Finite Elements.
    """
    
    def solve_covariance_EVP(cov, k):
        V = dolfin.function.functionspace.FunctionSpace(mesh, 'P', 1)
        u = dolfin.function.argument.TrialFunction(V)
        v = dolfin.function.argument.TestFunction(V)
    
        dof2vert = dolfin.cpp.fem.dof_to_vertex_map(V)
        coords = mesh.coordinates()
        coords = coords[dof2vert]
        M = dolfin.fem.assembling.assemble(u*v*dolfin.dx)
        M = M.array()
        L = coords.shape[0]
        
        if(distances is None):            
            c0 = np.repeat(coords, L, axis=0)
            c1 = np.tile(coords, [L,1])
            r = np.abs(np.linalg.norm(c0-c1, axis=1))
        else:
            r = distances
        C = cov(r)
        C.shape = [L,L]

        A = np.dot(M, np.dot(C, M))
        w, v = spla.eigsh(A, k, M)
        return w, v, V
    
    lambdas, phis, V  = solve_covariance_EVP(lambda r : covariance(r), k = nphis)
    inv = [(nphis-i-1) for i in range(nphis)]
    basis = phis[:, inv]
    lamb = lambdas[inv]
    
    return lamb, basis

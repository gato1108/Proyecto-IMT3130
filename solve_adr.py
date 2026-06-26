
'''
Recordatorio: Quiero resolver, dado un f, la ecuación

-∆u + u = f

que tiene como forma débil

(∇u, ∇v) + (u, v) = (f, v)

para todo v

En particular, quiero saber como resolverlo cuando solo tengo f visto desde los nodos
'''

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem

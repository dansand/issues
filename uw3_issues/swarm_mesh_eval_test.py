import numpy as np
import os
import math
import underworld3 as uw
from petsc4py import PETSc




boxLength      = 1.0
boxHeight      = 1.0
n_els = 32
dim = 2
ppcell = 10
amplitude  = 0.02
offset     = 0.2


mesh = uw.mesh.Mesh(elementRes=(    n_els,)*dim,
                    minCoords =(       0.,)*dim,
                    maxCoords =(boxLength,1.),
                    simplex=False )
u_degree = 1


# Create swarm
swarm  = uw.swarm.Swarm(mesh)
# Add variable for material
matSwarmVar      = swarm.add_variable(name="matSwarmVar",      num_components=1, dtype=PETSc.IntType)
# Note that `ppcell` specifies particles per cell per dim.
swarm.populate(ppcell=ppcell)


#%%
# define these for convenience.
denseIndex = 0
lightIndex = 1

# material perturbation from van Keken et al. 1997
wavelength = 2.0*boxLength
k = 2. * np.pi / wavelength

# init material variable
with swarm.access(matSwarmVar):
    perturbation = offset + amplitude*np.cos( k*swarm.particle_coordinates.data[:,0] )
    matSwarmVar.data[:,0] = np.where( perturbation>swarm.particle_coordinates.data[:,1], lightIndex, denseIndex )




from sympy import Piecewise, ceiling, Abs

density = Piecewise( ( 0., Abs(matSwarmVar.fn - lightIndex)<0.5 ),
                     ( 1., Abs(matSwarmVar.fn - denseIndex)<0.5 ),
                     ( 0.,                                True ) )



#this should produce an error
check_eval = uw.function.evaluate(density, mesh.data)

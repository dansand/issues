
# coding: utf-8

# In[9]:


import underworld as uw
from underworld import function as fn
import glucifer
import numpy as np


# In[10]:


# Set simulation box size.
boxHeight = 1.0
boxLength = 3.142
# Set the resolution.
res = 64


# In[11]:

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                 elementRes  = (3*res, res), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (boxLength, boxHeight))

tWalls = mesh.specialSets["MaxJ_VertexSet"]


# In[12]:

coordinate = fn.input()
randomFn = 2.*fn.math.cos(coordinate[0])


_minmax = fn.view.min_max(randomFn, fn_auxiliary=coordinate)
dummyFn = _minmax.evaluate(tWalls)


# In[13]:

print(_minmax.max_global())
print(_minmax.max_global_auxiliary()[0][0])


# In[14]:

rtol = 1e-6

assert np.allclose( _minmax.max_global(), 2.0, rtol=rtol ), "Error occurred in computing global max"
assert np.allclose(_minmax.max_global_auxiliary()[0][0], 0.0, rtol=rtol ), "Error occurred in auxilliary"


# In[ ]:




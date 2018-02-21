
# coding: utf-8

# In[10]:


import underworld as uw
import glucifer
import numpy as np
from underworld import function as fn


# In[11]:


# 1st a mesh is required to define a numerical domain
mesh = uw.mesh.FeMesh_Cartesian( elementRes  = (24, 24), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (1., 1.))


# In[12]:


xFn = fn.input()[0]
yFn = fn.input()[1]


# In[13]:


tWalls = mesh.specialSets["MaxJ_VertexSet"]


# In[15]:


_minMax = fn.view.min_max(xFn , fn_auxiliary=yFn)
_minMax.reset()
_minMax.evaluate(tWalls);


# In[16]:


print(_minMax.min_global())
print(_minMax.min_global_auxiliary())


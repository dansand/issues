
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


# In[19]:


# initialise a swarm
swarm = uw.swarm.Swarm( mesh=mesh )




# In[29]:


tWalls = mesh.specialSets["MaxJ_VertexSet"]

swarmCoords = mesh.data[tWalls.data]
swarm.add_particles_with_coordinates(swarmCoords)
#exploring a workaround


if not tWalls.data.shape[0]:
    dumCoord = np.column_stack((mesh.data[:,0].mean(), mesh.data[:,1].mean()))
    swarm.add_particles_with_coordinates(np.array(dumCoord))


# In[27]:


_minMax = fn.view.min_max(xFn , fn_auxiliary=yFn)
_minMax.reset()
_minMax.evaluate(swarm);


# In[28]:


print(_minMax.min_global())
print(_minMax.min_global_auxiliary())


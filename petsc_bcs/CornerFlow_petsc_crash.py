
# coding: utf-8

# In[141]:

import numpy as np
import underworld as uw
import math
from underworld import function as fn
import glucifer
import os
import sys
from easydict import EasyDict as edict
import operator
import pickle


# In[142]:

#sys.path


# #this does't actually need to be protected. More a reminder it's an interim measure
# try:
#     sys.path.append('../unsupported')
#     sys.path.append('./unsupported')
# except:
#     pass

# #
# from unsupported_dan.utilities.interpolation import nn_evaluation
# from unsupported_dan.interfaces.marker2D import markerLine2D
# from unsupported_dan.faults.faults2D import fault2D, fault_collection
# 

# ## Setup out Dirs

# In[145]:

############
#Model letter and number
############


#Model letter identifier default
Model = "T"

#Model number identifier default:
ModNum = 1

#Any isolated letter / integer command line args are interpreted as Model/ModelNum

if len(sys.argv) == 1:
    ModNum = ModNum 
elif sys.argv[1] == '-f': #
    ModNum = ModNum 
else:
    for farg in sys.argv[1:]:
        if not '=' in farg: #then Assume it's a not a paramter argument
            try:
                ModNum = int(farg) #try to convert everingthing to a float, else remains string
            except ValueError:
                Model  = farg


# In[146]:

###########
#Standard output directory setup
###########

outputPath = "results" + "/" +  str(Model) + "/" + str(ModNum) + "/" 
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
#checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
xdmfPath = outputPath + 'xdmf/'
outputFile = 'results_model' + Model + '_' + str(ModNum) + '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    if not os.path.isdir(xdmfPath):
        os.makedirs(xdmfPath)
        
uw.barrier() #Barrier here so no procs run the check in the next cell too early


# ## Params

# In[147]:

dp = edict({})
#Main physical paramters
dp.depth=300e3                         #Depth
dp.refDensity=3300.                        #reference density
dp.refGravity=9.8                          #surface gravity
dp.viscosityScale=1e20                       #reference upper mantle visc., 
dp.refDiffusivity=1e-6                     #thermal diffusivity
dp.refExpansivity=3e-5                     #surface thermal expansivity
dp.gasConstant=8.314                    #gas constant
dp.specificHeat=1250.                   #Specific heat (Jkg-1K-1)
dp.potentialTemp=1573.                  #mantle potential temp (K)
dp.surfaceTemp=273.                     #surface temp (K)
#Rheology - flow law paramters
dp.cohesionMantle=20e6                   #mantle cohesion in Byerlee law
dp.cohesionInterface=2e6                    #crust cohesion in Byerlee law
dp.frictionMantle=0.2                   #mantle friction coefficient in Byerlee law (tan(phi))
dp.frictionInterface=0.02                   #crust friction coefficient 
dp.diffusionPreExp=5.34e-10             #1./1.87e9, pre-exp factor for diffusion creep
dp.diffusionEnergy=3e5 
dp.diffusionVolume=5e-6

#
dp.interfacePreExp=2e2*5.34e-10            
dp.interfaceEnergy=0.4*3e5
dp.interfaceVolume=5.*5e-6

#power law creep params
dp.powerLawStrain = 1e-15
dp.powerLawExp = 3.5


#Rheology - cutoff values
dp.viscosityMin=1e18
dp.viscosityMax=1e25                #viscosity max in the mantle material
dp.viscosityMinInterface=1e20               #viscosity min in the weak-crust material
dp.viscosityMaxInterface=1e20               #viscosity max in the weak-crust material
dp.yieldStressMax=300*1e6              #

#Intrinsic Lengths
dp.faultThickness = 10*1e3              #interface material (crust) an top of slabs
dp.leftSide=-1.*(150e3)               #
dp.rightSide=(150e3)
dp.theta=45.                             #Angle of slab
dp.radiusOfCurv = 250e3                          #radius of curvature
dp.slabAge=70e6                     #age of subduction plate at trench
dp.opAge=35e6                       #age of op
dp.subZoneLoc=dp.leftSide                    #X position of subduction zone...km
dp.subVelocity = 4*(1/100.)*(1./(3600*24*365)) #m/s


#derived params
dp.deltaTemp = dp.potentialTemp-dp.surfaceTemp
dp.tempGradMantle = (dp.refExpansivity*dp.refGravity*(dp.potentialTemp))/dp.specificHeat
dp.tempGradSlab = (dp.refExpansivity*dp.refGravity*(dp.surfaceTemp + 400.))/dp.specificHeat



#Modelling and Physics switches

md = edict({})
md.refineMeshStatic=True
md.stickyAir=False
md.aspectRatio=1.
md.res=48
md.ppc=35                                 #particles per cell
#md.elementType="Q1/dQ0"
md.elementType="Q2/DPC1"
md.courantFac=0.5                         #extra limitation on timestepping
md.nltol = 0.001
md.maxSteps = 2000
md.druckerAlpha = 0.
md.druckerAlphaFault = 0.
md.penaltyMethod = True
md.spuniform = False
md.dissipativeHeating = True
md.powerLaw = False
md.interfaceDiffusivityFac = 1.



# In[150]:

sf = edict({})

sf.lengthScale=300e3
sf.viscosityScale = dp.viscosityScale
sf.stress = (dp.refDiffusivity*sf.viscosityScale)/sf.lengthScale**2
sf.lithGrad = dp.refDensity*dp.refGravity*(sf.lengthScale)**3/(sf.viscosityScale*dp.refDiffusivity) 
sf.lithGrad = (sf.viscosityScale*dp.refDiffusivity) /(dp.refDensity*dp.refGravity*(sf.lengthScale)**3)
sf.velocity = dp.refDiffusivity/sf.lengthScale
sf.strainRate = dp.refDiffusivity/(sf.lengthScale**2)
sf.time = 1./sf.strainRate
sf.actVolume = (dp.gasConstant*dp.deltaTemp)/(dp.refDensity*dp.refGravity*sf.lengthScale)
sf.actEnergy = (dp.gasConstant*dp.deltaTemp)
sf.diffusionPreExp = 1./sf.viscosityScale
sf.deltaTemp  = dp.deltaTemp
sf.pressureDepthGrad = (dp.refDensity*dp.refGravity*sf.lengthScale**3)/(dp.viscosityScale*dp.refDiffusivity)


#sf.dislocationPreExp = ((dp.refViscosity**(-1.*dp.dislocationExponent))*(dp.refDiffusivity**(1. - dp.dislocationExponent))*(sf.lengthScale**(-2.+ (2.*dp.dislocationExponent)))),
#sf.peierlsPreExp = 1./2.6845783276046923e+40 #same form as Ads, but ndp.np =20. (hardcoded because numbers are too big)


#dimesionless params
ndp  = edict({})

ndp.rayleigh = (dp.refExpansivity*dp.refDensity*dp.refGravity*dp.deltaTemp*sf.lengthScale**3)/(dp.viscosityScale*dp.refDiffusivity)
ndp.dissipation = (dp.refExpansivity*sf.lengthScale*dp.refGravity)/dp.specificHeat


#Take care with these definitions, 
ndp.surfaceTemp = dp.surfaceTemp/sf.deltaTemp  #Ts
ndp.potentialTemp = dp.potentialTemp/sf.deltaTemp - ndp.surfaceTemp #Tp' = Tp - TS

ndp.tempGradMantle = dp.tempGradMantle/(sf.deltaTemp/sf.lengthScale)
ndp.tempGradSlab = dp.tempGradSlab/(sf.deltaTemp/sf.lengthScale)

#lengths / distances
ndp.depth = dp.depth/sf.lengthScale
ndp.leftSide = dp.leftSide/sf.lengthScale             #
ndp.rightSide = dp.rightSide/sf.lengthScale
ndp.faultThickness = dp.faultThickness/sf.lengthScale



#times - for convenience the dimensional values are in years, conversion to seconds happens here
ndp.slabAge =  dp.slabAge*(3600*24*365)/sf.time
ndp.opAge = dp.opAge*(3600*24*365)/sf.time


#Rheology - flow law paramters
ndp.cohesionMantle=dp.cohesionMantle/sf.stress                  #mantle cohesion in Byerlee law
ndp.cohesionInterface=dp.cohesionInterface/sf.stress                  #crust cohesion in Byerlee law
ndp.frictionMantle=dp.frictionMantle/sf.lithGrad                  #mantle friction coefficient in Byerlee law (tan(phi))
ndp.frictionInterface=dp.frictionInterface/sf.lithGrad                  #crust friction coefficient 
ndp.diffusionPreExp=dp.diffusionPreExp/sf.diffusionPreExp                #pre-exp factor for diffusion creep
ndp.diffusionEnergy=dp.diffusionEnergy/sf.actEnergy
ndp.diffusionVolume=dp.diffusionVolume/sf.actVolume
#
ndp.interfacePreExp = dp.interfacePreExp/sf.diffusionPreExp           
ndp.interfaceEnergy = dp.interfaceEnergy/sf.actEnergy
ndp.interfaceVolume = dp.interfaceVolume/sf.actVolume

#
ndp.powerLawStrain = dp.powerLawStrain/sf.strainRate
ndp.powerLawExp = dp.powerLawExp

ndp.yieldStressMax=dp.yieldStressMax/sf.stress 


#Rheology - cutoff values
ndp.viscosityMin= dp.viscosityMin /sf.viscosityScale
ndp.viscosityMax=dp.viscosityMax/sf.viscosityScale
ndp.viscosityMinInterface= dp.viscosityMinInterface /sf.viscosityScale
ndp.viscosityMaxInterface = dp.viscosityMaxInterface/sf.viscosityScale




#Slab and plate init. parameters
ndp.subZoneLoc = dp.subZoneLoc/sf.lengthScale
ndp.radiusOfCurv = dp.radiusOfCurv/sf.lengthScale
ndp.theta=dp.theta #Angle of slab
ndp.subVelocity = dp.subVelocity/sf.velocity



# In[ ]:




# ## Make mesh / FeVariables

# In[151]:

#1. - ndp.depth


# In[152]:

#Domain and Mesh paramters
yres = int(md.res)
xres = int(md.res*md.aspectRatio) 



mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (xres, yres), 
                                 minCoord    = (ndp.leftSide, 1. - ndp.depth), 
                                 maxCoord    = (ndp.rightSide, 1.)) 

velocityField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
initialtemperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

stressField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=3 )


temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 ) #create this only if Adv-diff
diffusivityFn = fn.misc.constant(1.)
    


# In[153]:

velocityField.data[:] = 0.
pressureField.data[:] = 0.
temperatureField.data[:] = 0.
initialtemperatureField.data[:] = 0.


# In[154]:

#Uw geometry shortcuts

coordinate = fn.input()
depthFn = mesh.maxCoord[1] - coordinate[1] #a function providing the depth


xFn = coordinate[0]  #a function providing the x-coordinate
yFn = coordinate[1]


# ## Temp. Field

# In[155]:

temperatureField.data[:] = 1


# ## Boundary Conditions

# In[156]:

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
tWalls = mesh.specialSets["MaxJ_VertexSet"]
bWalls =mesh.specialSets["MinJ_VertexSet"]
lWalls = mesh.specialSets["MinI_VertexSet"]
rWalls = mesh.specialSets["MaxI_VertexSet"]
      
        


# In[157]:

#Now we need to set the velocity in the Slab 


# In[158]:

#use a circle Fn to tag some nodes for a velocity condition

def circleFn(centre, radius):
    _circFn = (((coordinate[0] - centre[0])**2) + ((coordinate[1] - centre[1])**2)) < radius**2
    return _circFn

circ1 = circleFn((-0.25, 0.8), 0.05)
nodes = circ1.evaluate(mesh).nonzero()[0]


# In[171]:

nodes = circ1.evaluate(mesh).nonzero()[0]

velocityField.data[nodes]= [1.0,0.]

drivenVel = mesh.specialSets["Empty"]

#if uw.rank() == 0:
drivenVel.add(nodes)  

drivenVel = drivenVel - lWalls - bWalls - rWalls


# In[160]:

#All the bCs

stressField.data[...] = (0.0,0.0,0.0)


velDbc = uw.conditions.DirichletCondition( variable      = velocityField, 
                                               indexSetsPerDof = ( iWalls + drivenVel, jWalls + drivenVel) )




# In[161]:

viscosityMapFn = fn.misc.constant(1.)


# In[162]:

uw.barrier()


# ## Stokes

# In[163]:

stokesPIC = uw.systems.Stokes( velocityField  = velocityField, 
                                   pressureField  = pressureField,
                                   conditions     = [velDbc,],
                                   fn_viscosity   = viscosityMapFn, 
                                   fn_bodyforce   = (0., 0.) )


# In[164]:

#md.penaltyMethod=False


# In[173]:

solver = uw.systems.Solver(stokesPIC)


    
    
    



# In[ ]:

solver.set_inner_method("mumps")
#solver.ptions.scr.ksp_type="cg"
solver.set_penalty(1.0e5)



# In[167]:

print("First solve")


# In[168]:

#solver.solve(nonLinearIterate=True, nonLinearTolerance=md.nltol)
solver.solve()


# In[169]:

uw.barrier()


# In[170]:

print("solve done")


# In[178]:

solver.options.scr.ksp_type


# In[ ]:




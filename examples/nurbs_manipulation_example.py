"""
nurbs_manipulation_example.py

# Example on how one can use nurbs mapping and prolongate it in fine grid

@author : M. BAHARI
"""

from   simplines                    import SplineSpace
from   simplines                    import TensorSpace
# from   simplines                    import sol_field_NURBS_2d
# from   simplines                    import prolongate_NURBS_mapping
from   simplines                    import paraview_nurbsSolutionMultipatch
from   simplines                    import getGeometryMap

#...
import numpy as np
#==============================================
#==============================================
nbpts    = 100
RefParm  = 2
# geometry = '../fields/quart_annulus.xml'
# geometry = '../fields/circle.xml'
# geometry  = '../fields/egg.xml'
geometry  = '../fields/cylinder.xml'

print('#---IN-UNIFORM--MESH-Poisson equation', geometry)
mp              = getGeometryMap(geometry,0)
print("geom dim = ",mp.geo_dim)

#==============================================
#... prolongate nurbs mapping
#==============================================
# mx, my, wm1, wm2 = prolongate_NURBS_mapping(VH, Vh, w, (px, py))
if mp.geo_dim == 2 :
    weight, mx, my  = mp.RefineGeometryMap(Nelements=(RefParm*mp.nelements[0],RefParm*mp.nelements[1]))
    wm1, wm2 = weight[:,0], weight[0,:]
    # print(weight)
    # print(wm1, wm2,"\n .",  mx[1,:], "\n .",my[1,:])
else :
    weight, mx, my, mz  = mp.RefineGeometryMap(Nelements=(RefParm*mp.nelements[0],RefParm*mp.nelements[1]))
    # mx, my, mz  = mp.coefs()
    # weight      = mp.weights.reshape(mp.nbasis)  
    # print(weight)
    wm1         = weight[:,0] 
    wm2         = weight[0,:] 

    # print(wm1, wm2,"\n .",  mx[3,:], "\n .",my[3,:], "\n .",mz[3,:])
#==============================================
#... spaces
#==============================================
grids      = []
for i in range(0, mp.nelements[0]):
    a = (mp.grids[0][i+1] - mp.grids[0][i])/RefParm
    grids.append(mp.grids[0][i])
    if a != 0. :
        for j in range(1,RefParm):
            grids.append(grids[-1] + a)
grids.append(mp.grids[0][-1])

V1         = SplineSpace(degree = mp.degree[0],  grid= grids, omega = wm1)
grids      = mp.knots[1][mp.degree[1]:-mp.degree[1]]
grids      = []
for i in range(0, mp.nelements[1]):
    a = (mp.grids[1][i+1] - mp.grids[1][i])/RefParm
    grids.append(mp.grids[1][i])
    if a != 0. :
        for j in range(1,RefParm):
            grids.append(grids[-1] + a)
grids.append(mp.grids[1][-1])
V2         = SplineSpace(degree = mp.degree[1],  grid= grids, omega = wm2)
Vh         = TensorSpace(V1, V2)

# ... save a solution as .vtm for paraview
if mp.geo_dim == 2:
    paraview_nurbsSolutionMultipatch(nbpts, [Vh], [mx], [my])
else:
    paraview_nurbsSolutionMultipatch(nbpts, [Vh], [mx], [my], zmp = [mz])
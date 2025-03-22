import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from scipy.sparse import kron, csr_matrix

from .cad import point_on_bspline_curve
from .cad import point_on_bspline_surface
from .bsplines import hrefinement_matrix


__all__ = ['plot_field_1d', 'plot_field_2d', 'prolongation_matrix']

# ==========================================================
def plot_field_1d(knots, degree, u, nx=101, color='b', xmin = None, xmax = None, label = None):
    n = len(knots) - degree - 1

    if xmin is None :
        xmin = knots[degree]
    if xmax is None :
        xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(u), 1))
    P[:,0] = u[:]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    if label is not None :
        plt.plot(xs, Q[:,0], label = label)
    else :
        plt.plot(xs, Q[:,0])

# ==========================================================
def plot_field_2d(knots, degrees, u, nx=101, ny=101):
    T1,T2 = knots
    p1,p2 = degrees

    n1 = len(T1) - p1 - 1
    n2 = len(T2) - p2 - 1

    xmin = T1[p1]
    xmax = T1[-p1-1]

    ymin = T2[p2]
    ymax = T2[-p2-1]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    n1,n2 = u.shape

    P = np.zeros((n1, n2, 1))
    P[:,:,0] = u[:,:]
    Q = np.zeros((nx, ny, 1))
    for i1,x in enumerate(xs):
        for i2,y in enumerate(ys):
            Q[i1,i2,:] = point_on_bspline_surface(T1, T2, P, x, y)
    X,Y = np.meshgrid(xs,ys)
    plt.contourf(X, Y, Q[:,:,0])

# ==========================================================
def prolongation_matrix(VH, Vh):
    # TODO not working for duplicated internal knots

    # ... TODO check that VH is included in Vh
    # ...

    # ...
    mats = []
    for Wh, WH in zip(Vh.spaces, VH.spaces):
        ths = Wh.knots
        tHs = WH.knots
        ts = set(ths) - set(tHs)
        ts = np.array(list(ts))

        M = hrefinement_matrix( ts, Wh.degree, tHs )
        mats.append(csr_matrix(M))
    # ...

    M = reduce(kron, (m for m in mats))

    return M


from .linalg      import StencilVector
from .spaces      import SplineSpace
from .spaces      import TensorSpace
from .results_f90 import least_square_Bspline
from .results_f90 import pyccel_sol_field_2d
from numpy import exp
from numpy import cos
from numpy import sin
from numpy import pi
from numpy import arctan2
from numpy import sqrt
from numpy import cosh
from numpy import zeros
from numpy import empty
def build_dirichlet(V, f, map = None, admap = None):
    '''
    V    : FE space
    f[0] : on the left
    f[1] : on the right
    f[2] : on the bottom
    f[3] : on the top
    map = (x,y) : control points
    admap = (x, V1, y, V2) control points and associated space
    '''
    if len(f) > 1 :
        fx0      = lambda x,y :  eval(f[0])
        fx1      = lambda x,y :  eval(f[1])
        fy0      = lambda x,y :  eval(f[2])
        fy1      = lambda x,y :  eval(f[3])
    elif map is None:
        xi       = V.grid[0]
        yi       = V.grid[1]
        sol      = lambda x,y :  eval(f[0]) 
        fx0      = lambda   y :  sol(xi[0],y)
        fx1      = lambda   y :  sol(xi[-1],y) 
        fy0      = lambda x   :  sol(x,yi[0])
        fy1      = lambda x   :  sol(x,yi[-1])
    else :
        fx0      = lambda x,y :  eval(f[0])
        fx1      = lambda x,y :  eval(f[0])
        fy0      = lambda x,y :  eval(f[0])
        fy1      = lambda x,y :  eval(f[0]) 
    u_d          = StencilVector(V.vector_space)
    x_d          = np.zeros(V.nbasis)
    if V.dim == 2:
        if map is None:
            #------------------------------
            #.. In the parametric domain
            x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0)
            x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1)
            x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0)
            x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1)

        elif admap is None :
            #------------------------------
            #.. In the phyisacl domain without adaptive mapping               
            n_dir        = V.nbasis[0] + V.nbasis[1]+100
            sX           = pyccel_sol_field_2d((n_dir,n_dir),  map[0] , V.knots, V.degree)[0]
            sY           = pyccel_sol_field_2d((n_dir,n_dir),  map[1] , V.knots, V.degree)[0]

            x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0(sX[0, :], sY[ 0,:]), m= n_dir)
            x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1(sX[-1,:], sY[-1,:]), m= n_dir)
            x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0(sX[:, 0], sY[:, 0]), m= n_dir)
            x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1(sX[:,-1], sY[:,-1]), m= n_dir)

        else :
            #------------------------------
            #.. In the phyisacl domain with adaptive mapping               
            n_dir        = V.nbasis[0] + V.nbasis[1]+100

            Xmae         = pyccel_sol_field_2d((n_dir,n_dir),  admap[0] , admap[2].knots, admap[2].degree)[0]
            Ymae         = pyccel_sol_field_2d((n_dir,n_dir),  admap[1] , admap[3].knots, admap[3].degree)[0]
            sX           = pyccel_sol_field_2d( None, map[0], V.knots, V.degree, meshes = (Xmae, Ymae))[0]
            sY           = pyccel_sol_field_2d( None, map[1], V.knots, V.degree, meshes = (Xmae, Ymae))[0]

            x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0(sX[0, :], sY[ 0,:]), m= n_dir)
            x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1(sX[-1,:], sY[-1,:]), m= n_dir)
            x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0(sX[:, 0], sY[:, 0]), m= n_dir)
            x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1(sX[:,-1], sY[:,-1]), m= n_dir)
    if V.dim == 3 :
        print("not yet implemented")
        #.. TODO : USE l2 PROJECTION USING FAST DIAG
    u_d.from_array(V, x_d)
    return x_d, u_d

import xml.etree.ElementTree as ET
import numpy as np

def save_geometry_to_xml(V, Gmap, name = 'Geometry', locname = None):
    """
    save_geometry_to_xml : save the coefficients table, knots table, and degree in an XML file.
    """
    if locname is None :
        filename = f'./figs/'+name+'.xml'
    else :
        filename = locname+'.xml'
    # Root element
    root = ET.Element('xml')
    root.text  = '\n'
    
    # Geometry element
    geometry    = ET.SubElement(root, 'Geometry', type='TensorNurbs2', id='0')
    geometry.text = '\n'
    basis_outer = ET.SubElement(geometry, 'Basis', type='TensorNurbsBasis2')
    basis_outer.text = '\n'
    basis_inner = ET.SubElement(basis_outer, 'Basis', type='TensorBSplineBasis2')
    basis_inner.text = '\n'

    # Add basis elements
    for i in range(2):
        basis            = ET.SubElement(basis_inner, 'Basis', type='BSplineBasis', index=str(i))
        basis.text       = '\n'
        knot_vector      = ET.SubElement(basis, 'KnotVector', degree=str(V.degree[0]))
        knot_vector.text = '\n' + ' '.join(map(str, V.knots[i])) + '\n'
    
    # Add coefficients (control points)
    coefs      = ET.SubElement(basis_inner, 'coefs', geoDim='2')
    coefs.text = '\n' + '\n'.join(' '.join(f'{v:.20e}' for v in row) for row in Gmap) + '\n'

    # Close inner Basis element properly
    basis_inner.tail = '\n'
    basis_outer.tail = '\n'

    # MultiPatch element
    multipatch   = ET.SubElement(root, 'MultiPatch', parDim='2', id='1')
    multipatch.text = '\n'
    patches      = ET.SubElement(multipatch, 'patches', type='id_range')
    patches.text = '\n0 0\n'
    
    # Boundary conditions
    boundary      = ET.SubElement(multipatch, 'boundary')
    boundary.text = '\n  0 1\n  0 2\n  0 3\n  0 4\n '
    boundary.tail = '\n'
    
    # Convert to XML string with declaration
    xml_string = ET.tostring(root, encoding='utf-8').decode('utf-8')
    xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' \
                 '<!--This file was created by bahari95/simplines -->\n' \
                 '<!--Geometry in two dimensions -->\n' + xml_string
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_string)
    
    print(f"File saved as {filename}")

class getGeometryMap:
    """
    getGeometryMap : extracts the coefficients table, knots table, and degree from an XML file based on a given id.
    """
    def __init__(self, filename, element_id):
      #print("""Initialize with the XML filename.""", filename)
      root            = ET.parse(filename).getroot()
      """Retrieve coefs table, knots table, and degree for a given id."""
      # Find the Geometry element by id
      GeometryMap = root.find(f".//*[@id='{element_id}']")        
      if GeometryMap is None:
         raise RuntimeError(f"No element found with id {element_id}")

      # Extract knots data and degree
      knots_data  = []
      degree_data = []
      for basis in GeometryMap.findall(".//Basis[@type='BSplineBasis']"):
         knot_vector = basis.find("KnotVector")
         if knot_vector is not None:
               degree_data.append(int(knot_vector.get("degree", -1)))  # Default to -1 if not found
               knots = list(map(float, knot_vector.text.strip().split()))
               knots_data.append(knots)
      #....dimension
      dim              = np.asarray(knots_data).shape[0]
      #....number of basis functions
      nbasis           = [len(np.asarray(knots_data)[n,:]) - degree_data[n]-1 for n in range(dim)]
      # Extract coefs data
      coefs_element = GeometryMap.find(".//coefs")
      coefs_data    = None
      if coefs_element is not None:
         coefs_text = coefs_element.text.strip()
         coefs_data = np.array([
               list(map(float, line.split())) for line in coefs_text.split("\n")
         ])

      self.root        = root
      self.GeometryMap = GeometryMap
      self.knots_data  = knots_data
      self._degree     = degree_data
      self._coefs      = [coefs_data[:,n].reshape(nbasis) for n in range(dim)]
      self._dim        = dim
      self._nbasis     = nbasis
      self._nelements  = [nbasis[n]-degree_data[n] for n in range(dim)]

    @property
    def nbasis(self):
        return self._nbasis
    @property
    def dim(self):
        return self._dim
    @property
    def knots(self):
        return self.knots_data
    @property
    def degree(self):
        return self._degree
    @property
    def nelements(self):
        return self._nelements
    def coefs(self):
        return self._coefs   
    
    def RefineGeometryMap(self, numElevate=0, Nelements=None):
        """
        getGeometryMap :  Refine the geometry by elevating the DoFs numElevate times.
        """
        assert(numElevate >= 0)
        #... refine the grid numElevate times
        if Nelements is None:
            Nelements = [self.nelements[n]**numElevate for n in range(self.dim)]
        else :
            assert(len(Nelements) == self.dim and Nelements[0]%self.nelements[0]==0 and Nelements[1]%self.nelements[1]==0)
        #... refine the space
        Vh1       = SplineSpace(degree=self.degree[0], nelements=Nelements[0])
        Vh2       = SplineSpace(degree=self.degree[1], nelements=Nelements[1])
        Vh        = TensorSpace(Vh1, Vh2)# after refinement
        # Extract knots data and degree
        VH1       = SplineSpace(degree=self.degree[0], nelements=self.nelements[0])
        VH2       = SplineSpace(degree=self.degree[1], nelements=self.nelements[1])
        VH        = TensorSpace(VH1, VH2)# after refinement
        # Extract coefs data
        coefs_data = self.coefs()
        # Refine the coefs
        M_mp      = prolongation_matrix(VH, Vh)
        coefs_data[0]      = (M_mp.dot(coefs_data[0].reshape(self.nbasis[0]*self.nbasis[1]))).reshape(Vh.nbasis)
        coefs_data[1]      = (M_mp.dot(coefs_data[1].reshape(self.nbasis[0]*self.nbasis[1]))).reshape(Vh.nbasis)        
        return coefs_data
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from scipy.sparse import kron, csr_matrix

from .cad import point_on_bspline_curve
from .cad import point_on_bspline_surface
from .bsplines import hrefinement_matrix


__all__ = ['plot_field_1d', 
           'plot_field_2d', 
           'prolongation_matrix',
           'save_geometry_to_xml',
           'getGeometryMap']

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


from .linalg          import StencilVector
from .spaces          import SplineSpace
from .spaces          import TensorSpace
from .results_f90     import least_square_Bspline
from .results_f90     import pyccel_sol_field_2d
from .nurbs_utilities import sol_field_NURBS_2d
from .nurbs_utilities import least_square_NURBspline

from numpy            import exp
from numpy            import cos
from numpy            import sin, sinh
from numpy            import pi
from numpy            import arctan2
from numpy            import sqrt
from numpy            import cosh, tanh
from numpy            import zeros
from numpy            import empty
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
        fx0      = lambda   y :  eval(f[0])
        fx1      = lambda   y :  eval(f[1])
        fy0      = lambda x   :  eval(f[2])
        fy1      = lambda x   :  eval(f[3])
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
    if V.omega[0] is None :
        #... B-spline space
        if V.dim == 2:
            if map is None:
                #------------------------------
                #.. In the parametric domain
                x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0)
                x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1)
                x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0)
                x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1)

            elif admap is None :
                #-------------------------------------------------
                #.. In the phyisacl domain without adaptive mapping               
                n_dir        = V.nbasis[0] + V.nbasis[1]+100
                sX           = pyccel_sol_field_2d((n_dir,n_dir),  map[0] , V.knots, V.degree)[0]
                sY           = pyccel_sol_field_2d((n_dir,n_dir),  map[1] , V.knots, V.degree)[0]

                x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0(sX[0, :], sY[ 0,:]))
                x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1(sX[-1,:], sY[-1,:]))
                x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0(sX[:, 0], sY[:, 0]))
                x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1(sX[:,-1], sY[:,-1]))

            else :
                #-----------------------------------------------
                #.. In the phyisacl domain with adaptive mapping               
                n_dir        = V.nbasis[0] + V.nbasis[1]+100

                Xmae         = pyccel_sol_field_2d((n_dir,n_dir),  admap[0] , admap[2].knots, admap[2].degree)[0]
                Ymae         = pyccel_sol_field_2d((n_dir,n_dir),  admap[1] , admap[3].knots, admap[3].degree)[0]
                sX           = pyccel_sol_field_2d( None, map[0], V.knots, V.degree, meshes = (Xmae, Ymae))[0]
                sY           = pyccel_sol_field_2d( None, map[1], V.knots, V.degree, meshes = (Xmae, Ymae))[0]

                x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0(sX[0, :], sY[ 0,:]))
                x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1(sX[-1,:], sY[-1,:]))
                x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0(sX[:, 0], sY[:, 0]))
                x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1(sX[:,-1], sY[:,-1]))
        if V.dim == 3 :
            raise NotImplementedError("3D Dirichlet boundary conditions are not yet implemented. nd: Use L2 projection using fast diagonalization.")
    else :
        if V.dim == 2:
            if map is None:
                #------------------------------
                #.. In the parametric domain
                x_d[ 0 , : ] = least_square_NURBspline(V.degree[1], V.knots[1], V.omega[1], fx0)
                x_d[ -1, : ] = least_square_NURBspline(V.degree[1], V.knots[1], V.omega[1], fx1)
                x_d[ : , 0 ] = least_square_NURBspline(V.degree[0], V.knots[0], V.omega[0], fy0)
                x_d[ : ,-1 ] = least_square_NURBspline(V.degree[0], V.knots[0], V.omega[0], fy1)

            elif admap is None :
                #-------------------------------------------------
                #.. In the phyisacl domain without adaptive mapping               
                n_dir        = V.nbasis[0] + V.nbasis[1]+100
                sX           = sol_field_NURBS_2d((n_dir,n_dir),  map[0], V.omega, V.knots, V.degree)[0]
                sY           = sol_field_NURBS_2d((n_dir,n_dir),  map[1], V.omega, V.knots, V.degree)[0]

                x_d[ 0 , : ] = least_square_NURBspline(V.degree[1], V.knots[1], V.omega[1], fx0(sX[0, :], sY[ 0,:]))
                x_d[ -1, : ] = least_square_NURBspline(V.degree[1], V.knots[1], V.omega[1], fx1(sX[-1,:], sY[-1,:]))
                x_d[ : , 0 ] = least_square_NURBspline(V.degree[0], V.knots[0], V.omega[0], fy0(sX[:, 0], sY[:, 0]))
                x_d[ : ,-1 ] = least_square_NURBspline(V.degree[0], V.knots[0], V.omega[0], fy1(sX[:,-1], sY[:,-1]))

            else :
                #-----------------------------------------------
                #.. In the phyisacl domain with adaptive mapping               
                n_dir        = V.nbasis[0] + V.nbasis[1]+100

                Xmae         = sol_field_NURBS_2d((n_dir,n_dir),  admap[0], V.omega, V.knots, V.degree)[0]
                Ymae         = sol_field_NURBS_2d((n_dir,n_dir),  admap[1], V.omega, V.knots, V.degree)[0]
                sX           = sol_field_NURBS_2d( None, map[0], V.knots, V.degree, meshes = (Xmae, Ymae))[0]
                sY           = sol_field_NURBS_2d( None, map[1], V.knots, V.degree, meshes = (Xmae, Ymae))[0]

                x_d[ 0 , : ] = least_square_NURBspline(V.degree[1], V.knots[1], V.omega[1], fx0(sX[0, :], sY[ 0,:]))
                x_d[ -1, : ] = least_square_NURBspline(V.degree[1], V.knots[1], V.omega[1], fx1(sX[-1,:], sY[-1,:]))
                x_d[ : , 0 ] = least_square_NURBspline(V.degree[0], V.knots[0], V.omega[0], fy0(sX[:, 0], sY[:, 0]))
                x_d[ : ,-1 ] = least_square_NURBspline(V.degree[0], V.knots[0], V.omega[0], fy1(sX[:,-1], sY[:,-1]))
        if V.dim == 3 :
            raise NotImplementedError("3D Dirichlet boundary conditions are not yet implemented. nd: Use L2 projection using fast diagonalization.")
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
    if isinstance(Gmap, (list, tuple)):
        for pid, gmap in enumerate(Gmap):
            geom = ET.SubElement(root, 'Geometry', type='TensorNurbs2', id=str(pid))
            geom.text = '\n'

            outer = ET.SubElement(geom, 'Basis', type='TensorNurbsBasis2'); outer.text = '\n'
            inner = ET.SubElement(outer, 'Basis', type='TensorBSplineBasis2'); inner.text = '\n'

            # mêmes nœuds / degrés pour chaque patch
            for d in range(2):
                b = ET.SubElement(inner, 'Basis', type='BSplineBasis', index=str(d)); b.text = '\n'
                kv = ET.SubElement(b, 'KnotVector', degree=str(V.degree[d]))
                kv.text = '\n' + ' '.join(map(str, V.knots[d])) + '\n'
                b.tail = '\n'

            co = ET.SubElement(inner, 'coefs', geoDim='2')
            co.text = '\n' + '\n'.join(
                ' '.join(f'{v:.20e}' for v in row) for row in gmap
            ) + '\n'
            co.tail = '\n'            

            inner.tail = '\n'; outer.tail = '\n'; geom.tail = '\n'

        # ---------- MultiPatch ----------
        mp = ET.SubElement(root, 'MultiPatch', parDim='2', id=str(len(Gmap))); mp.text = '\n'
        pr = ET.SubElement(mp, 'patches', type='id_range')
        pr.text = f'\n0 {len(Gmap)-1}\n'
        bd = ET.SubElement(mp, 'boundary')
        bd.text = '\n' + ''.join(f'  {i} 1\n  {i} 2\n  {i} 3\n  {i} 4\n' for i in range(len(Gmap)))
        bd.tail = '\n'

    else :
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
      dim              = len(knots_data)
      #....number of basis functions
      _nbasis           = [len(knots_data[n]) - degree_data[n]-1 for n in range(dim)]
      # Extract coefs data
      coefs_element = GeometryMap.find(".//coefs")
      geo_dim       = int(coefs_element.attrib.get("geoDim")) if coefs_element is not None else None
      coefs_data    = None
      if coefs_element is not None:
         coefs_text = coefs_element.text.strip()
         coefs_data = np.array([
               list(map(float, line.split())) for line in coefs_text.split("\n")
         ])
      # Extract weights data or default to ones
      weights_element = GeometryMap.find(".//weights")
      nurbs_check     = False 
      if weights_element is not None:
          nurbs_check     = True
          weights_text = weights_element.text.strip()
          weights_data = np.array([
              float(line.strip()) for line in weights_text.split("\n") if line.strip()
          ])
      else:
          weights_data = np.ones(len(coefs_data[:,0]))

      self.root        = root
      self.GeometryMap = GeometryMap
      self.knots_data  = knots_data
      self._degree     = degree_data
      self._coefs      = np.asarray([coefs_data[:,n].reshape(_nbasis) for n in range(geo_dim)])
      self._grids      = [knots_data[n][degree_data[n]:-degree_data[n]] for n in range(dim)]
      self._weights    = weights_data
      self._dim        = dim
      self.nurbs_check = nurbs_check
      self._geo_dim    = geo_dim
      self._nbasis     = _nbasis
      self._nelements  = [_nbasis[n]-degree_data[n] for n in range(dim)]

    @property
    def nbasis(self):
        return self._nbasis
    @property
    def dim(self):
        return self._dim
    @property
    def geo_dim(self):
        return self._geo_dim
    @property
    def knots(self):
        return self.knots_data
    @property
    def degree(self):
        return self._degree
    @property
    def nelements(self):
        return self._nelements
    @property
    def grids(self):
        return self._grids
    @property
    def weights(self):
        return self._weights
    
    def coefs(self):
        return self._coefs      
    
    def Refinegrid(self, j_direct, Nelements, numElevate = 1):
        assert(numElevate >= 1)
        #... refine the grid numElevate times
        if Nelements is None:
            Nelements = [self.nelements[n]*numElevate for n in range(self.dim)]
        else :
            assert(len(Nelements) == self.dim and Nelements[0]%self.nelements[0]==0 and Nelements[1]%self.nelements[1]==0)
        grids      = []
        numElevate = Nelements[j_direct]//self.nelements[j_direct]
        for i in range(0, self.nelements[j_direct]):
            a = (self._grids[j_direct][i+1] - self._grids[j_direct][i])/numElevate
            grids.append(self._grids[j_direct][i])
            if a != 0. :
                for j in range(1,numElevate):
                    grids.append(grids[-1] + a)
        grids.append(self._grids[j_direct][-1])
        return grids
    
    def RefineGeometryMap(self, numElevate=1, Nelements=None):
        """
        getGeometryMap :  Refine the geometry by elevating the DoFs numElevate times.
        """
        assert(numElevate >= 1)
        #... refine the grid numElevate times
        if Nelements is None:
            Nelements = [self.nelements[n]*numElevate for n in range(self.dim)]
        else :
            assert(len(Nelements) == self.dim and Nelements[0]%self.nelements[0]==0 and Nelements[1]%self.nelements[1]==0)
        #... refine the space
        if self.dim == 2:
            grids     = self.Refinegrid(0, Nelements)
            Vh1       = SplineSpace(degree=self.degree[0], grid= grids)
            grids     = self.Refinegrid(1, Nelements)
            Vh2       = SplineSpace(degree=self.degree[1], grid= grids)
            Vh        = TensorSpace(Vh1, Vh2)# after refinement
            # Extract knots data and degree
            VH1       = SplineSpace(degree=self.degree[0], grid= self._grids[0])
            VH2       = SplineSpace(degree=self.degree[1], grid= self._grids[1])

            nbasis_tot = self._nbasis[0]*self._nbasis[1]
            VH         = TensorSpace(VH1, VH2)# after refinement
        else:
            grids      = self.Refinegrid(0, Nelements)
            Vh1        = SplineSpace(degree=self.degree[0], grid= grids)
            grids      = self.Refinegrid(1, Nelements)
            Vh2        = SplineSpace(degree=self.degree[1], grid= grids)
            grids      = self.Refinegrid(2, Nelements)
            Vh3        = SplineSpace(degree=self.degree[2], grid= grids)            
            Vh         = TensorSpace(Vh1, Vh2, Vh3)# after refinement
            # Extract knots data and degree
            #print(f"Refined space : {self.nelements[0]} x {self.nelements[1]} Nelements")
            VH1        = SplineSpace(degree=self.degree[0], grid= self._grids[0])
            VH2        = SplineSpace(degree=self.degree[1], grid= self._grids[1])
            VH3        = SplineSpace(degree=self.degree[2], grid= self._grids[2])
            VH         = TensorSpace(VH1, VH2, VH3)# after refinement
            nbasis_tot = self._nbasis[0]*self._nbasis[1]*self._nbasis[2]

        # Extract coefs data
        coefs_data = []
        # Refine the coefs
        M_mp      = prolongation_matrix(VH, Vh)
        if self.nurbs_check:
            coefs_data.append( (M_mp.dot(self._weights)).reshape(Vh.nbasis))
            for i in range(self.geo_dim):
                coefs_data.append( (M_mp.dot(self._weights*self.coefs()[i].reshape(nbasis_tot))).reshape(Vh.nbasis) / coefs_data[0])
            return coefs_data
        else:
            for i in range(self.geo_dim):
                coefs_data.append( (M_mp.dot(self.coefs()[i].reshape(nbasis_tot))).reshape(Vh.nbasis) )
            return coefs_data
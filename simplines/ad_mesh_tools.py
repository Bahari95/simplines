from   .linalg           import StencilVector
from   numpy             import zeros
from   numpy             import double
from   .                 import ad_mesh_core as core

class quadratures_in_admesh(object):
	'''
	The provided code calculates B-spline functions and their corresponding spans within the quadrature image using optimal mapping. 
	This mapping transforms a uniform mesh into an adaptive mesh within a unit square.
	'''
	def __init__(self, V, reparameterization = False, nders = 2) :
		# ...
		sp_dim = V.dim
		if reparameterization is True : 
			# ... L2(gradient) mapping
			self.basis_spans_in_adquadrature_2d = core.assemble_basis_spans_in_adquadrature_L2map
		elif sp_dim == 1 :
			# ... 1D reparametrization
			self.basis_spans_in_adquadrature_1d = core.assemble_basis_spans_in_adquadrature_1DL2map
		elif sp_dim == 2 :
			# ... gradient mapping
			self.basis_spans_in_adquadrature_2d = core.assemble_basis_spans_in_adquadrature_gradmap
		elif sp_dim == 3:
			#... L2 mapping in 3D
			self.basis_spans_in_adquadrature_3d = core.assemble_basis_spans_in_adquadrature_3L2map
		elif sp_dim == 6 :
			# ... The Hdiv mapping space can be selected independently of the initial mapping space.
			self.basis_spans_in_adquadrature_2d = core.assemble_basis_spans_in_adquadrature
		else :
			# ... L2-B-spline space (degree-1, degree-1) for Hdiv mapping is the same as for initial mapping.
			self.basis_spans_in_adquadrature_2d = core.assemble_basis_spans_in_adquadrature_same_space
		# ...
		if sp_dim == 1:
			# ...
			args  = [V.nelements,
					V.degree,
					V.spans,
					V.basis,
					V.weights,
					V.points,
					V.knots]
			#...
			p1       = V.degree
			nx       = V.nelements
			wx       = V.weights
			# ...
			k1       = wx.shape[1]
			# ...
			self.args    = args
			self.nb_vec  = (nx, p1+1, nders+1, k1)
			self.ns_vec  = (nx, k1)
		elif sp_dim == 3:
			# ...
			args  = []
			args += list(V.nelements)
			args += list(V.degree)
			args += list(V.spans)			
			args += list(V.basis)
			args += list(V.weights)
			args += list(V.points)
			args += list(V.knots)
			#...
			p1, p2, p3   = V.degree[-3:]
			nx, ny, nz   = V.nelements[-3:]
			wx, wy, wz   = V.weights[-3:]
			# ...
			k1           = wx.shape[1]
			k2           = wy.shape[1]
			k3           = wz.shape[1]
			# ...
			self.args    = args
			self.nb_vec0 = (nx, ny, nz, p1+1, nders+1, k1, k2, k3)
			self.nb_vec1 = (nx, ny, nz, p2+1, nders+1, k1, k2, k3)
			self.nb_vec2 = (nx, ny, nz, p3+1, nders+1, k1, k2, k3)
			self.ns_vec  = (nx, ny, nz, k1, k2, k3)
		else:
			# ... We test Hdiv and other solver in 2D
			args  = []
			args += list(V.nelements)
			args += list(V.degree)
			if sp_dim == 6 :
				args += list(V.spans[:-2])
			else :
				args += list(V.spans)			
			args += list(V.basis)
			args += list(V.weights)
			args += list(V.points)
			args += list(V.knots)
			#...
			p1, p2       = V.degree[-2:]
			nx, ny       = V.nelements[-2:]
			wx, wy       = V.weights[-2:]
			# ...
			k1           = wx.shape[1]
			k2           = wy.shape[1]
			# ...
			self.args    = args
			self.nb_vec0 = (nx, ny, p1+1, nders+1, k1, k2)
			self.nb_vec1 = (nx, ny, p2+1, nders+1, k1, k2)
			self.ns_vec  = (nx, ny, k1, k2)	
		self.nders   = nders
		self.sp_dim	 = sp_dim


	def ad_Gradmap_quadratures(self, u_mae):
		'''
		computes Quadratures using Gradient mapping
		'''
		# ...
		if isinstance(u_mae, StencilVector):
			# ...
			basis_ad1  = zeros(self.nb_vec0, dtype = double)
			basis_ad2  = zeros(self.nb_vec1, dtype = double)
			spans_ad1  = zeros(self.ns_vec, int)
			spans_ad2  = zeros(self.ns_vec, int)
			# ...
			self.basis_spans_in_adquadrature_2d(*self.args, u_mae._data, spans_ad1, spans_ad2, basis_ad1, basis_ad2, self.nders)			
			return spans_ad1, spans_ad2, basis_ad1, basis_ad2
		else :
			print("Error please Try with StencilVector")

	def ad_quadratures(self, u01_mae, u10_mae = None, u11_mae = None):
		'''
		computes Quadratures using Hdiv mapping
		'''
		if isinstance(u11_mae, StencilVector):
			# ... !3D
			basis_ad1  = zeros(self.nb_vec0, dtype = double)
			basis_ad2  = zeros(self.nb_vec1, dtype = double)
			basis_ad3  = zeros(self.nb_vec2, dtype = double)
			spans_ad1  = zeros(self.ns_vec, int)
			spans_ad2  = zeros(self.ns_vec, int)
			spans_ad3  = zeros(self.ns_vec, int)
			# ...
			self.basis_spans_in_adquadrature_3d(*self.args, u01_mae._data, u10_mae._data, u11_mae._data, spans_ad1, spans_ad2, spans_ad3, basis_ad1, basis_ad2, basis_ad3, self.nders)
			return spans_ad1, spans_ad2, spans_ad3, basis_ad1, basis_ad2, basis_ad3
		elif isinstance(u10_mae, StencilVector):
			# ... !2D and 2.5D
			basis_ad1  = zeros(self.nb_vec0, dtype = double)
			basis_ad2  = zeros(self.nb_vec1, dtype = double)
			spans_ad1  = zeros(self.ns_vec, int)
			spans_ad2  = zeros(self.ns_vec, int)
			# ...
			self.basis_spans_in_adquadrature_2d(*self.args, u01_mae._data, u10_mae._data, spans_ad1, spans_ad2, basis_ad1, basis_ad2, self.nders)
			return spans_ad1, spans_ad2, basis_ad1, basis_ad2

		elif self.sp_dim == 1 :
			# ... !1D
			basis_ad  = zeros(self.nb_vec, dtype = double)
			spans_ad  = zeros(self.ns_vec, int)
			# ...
			self.basis_spans_in_adquadrature_1d(*self.args, u01_mae._data, spans_ad, basis_ad, self.nders)
			return spans_ad, basis_ad
		else :
			print("Error please Try with StencilVector")

#... Some kernels for assembling matrices
from   functools import partial
from   .api      import assemble_matrix, assemble_vector
assemble_stiffness1D = partial(assemble_matrix, core.assemble_stiffnessmatrix1D)
assemble_mass1D      = partial(assemble_matrix, core.assemble_massmatrix1D)
assemble_matrix_ex01 = partial(assemble_vector, core.assemble_matrix_ex01)
assemble_matrix_ex02 = partial(assemble_vector, core.assemble_matrix_ex02)
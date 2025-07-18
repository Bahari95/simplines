from simplines import bsplines
from simplines import cad
from simplines import spaces
from simplines import linalg
from simplines import quadratures
from simplines import utilities
from simplines import api
from simplines import results
from simplines import results_f90
from simplines import ad_mesh_tools
from simplines import fast_diag
from simplines import nurbs_utilities

__all__ = ['bsplines', 'cad',
           'spaces', 'linalg',
           'quadratures', 'utilities', 'results', 'ad_mesh_tools', 'results_f90', 'api', 'nurbs_utilities']

from simplines.bsplines import ( find_span,
                                 basis_funs,
                                 basis_funs_1st_der,
                                 basis_funs_all_ders,
                                 collocation_matrix,
                                 histopolation_matrix,
                                 breakpoints,
                                 greville,
                                 elements_spans,
                                 make_knots,
                                 elevate_knots,
                                 quadrature_grid,
                                 basis_integrals,
                                 basis_ders_on_quad_grid,
                                 scaling_matrix,
                                 hrefinement_matrix )

from simplines.cad import ( point_on_bspline_curve,
                            point_on_nurbs_curve,
                            insert_knot_bspline_curve,
                            insert_knot_nurbs_curve,
                            elevate_degree_bspline_curve,
                            elevate_degree_nurbs_curve,
                            point_on_bspline_surface,
                            point_on_nurbs_surface,
                            insert_knot_bspline_surface,
                            insert_knot_nurbs_surface,
                            elevate_degree_bspline_surface,
                            elevate_degree_nurbs_surface,
                            translate_bspline_curve,
                            translate_nurbs_curve,
                            rotate_bspline_curve,
                            rotate_nurbs_curve,
                            homothetic_bspline_curve,
                            homothetic_nurbs_curve,
                            translate_bspline_surface,
                            translate_nurbs_surface,
                            homothetic_nurbs_curve,
                            translate_bspline_surface,
                            translate_nurbs_surface,
                            rotate_bspline_surface,
                            rotate_nurbs_surface,
                            homothetic_bspline_surface,
                            homothetic_nurbs_surface)

from simplines.spaces import ( SplineSpace,
                               TensorSpace )

from simplines.linalg import ( StencilVectorSpace,
                               StencilVector,
                               StencilMatrix )

from simplines.quadratures import gauss_legendre

from simplines.utilities import ( plot_field_1d,
                                  plot_field_2d,
                                  prolongation_matrix,
                                  build_dirichlet,
                                  getGeometryMap,
                                  save_geometry_to_xml)

from simplines.results import ( sol_field_2d)

from simplines.ad_mesh_tools import ( quadratures_in_admesh,
                                     assemble_stiffness1D,
                                     assemble_mass1D,
                                     assemble_matrix_ex01,
                                     assemble_matrix_ex02)

from simplines.fast_diag import ( Poisson)

from simplines.results_f90 import ( pyccel_sol_field_2d,
                                   pyccel_sol_field_1d,
                                    pyccel_sol_field_3d, 
                                    least_square_Bspline,
                                    plot_SolutionMultipatch,
                                    plot_MeshMultipatch,
                                    plot_AdMeshMultipatch,
                                    plot_FunctMultipatch,
                                    plot_JacobianMultipatch,
                                    paraview_AdMeshMultipatch,
                                    paraview_SolutionMultipatch)

from simplines.api import (assemble_matrix, assemble_vector, assemble_scalar, compile_kernel, apply_dirichlet, apply_periodic, apply_zeros)

from simplines.nurbs_utilities import(sol_field_NURBS_2d, sol_field_NURBS_3d, 
                                      prolongate_NURBS_mapping, least_square_NURBspline,
                                      paraview_nurbsAdMeshMultipatch, paraview_nurbsSolutionMultipatch,
                                      ViewGeo)
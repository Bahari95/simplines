"""
nurbs_manipulation_example.py

# Example on how one can show a geometry discrebed by nurbs or b-spline

@author : M. BAHARI
"""

from simplines import ViewGeo
#------------------------------------------------------------------------------
# Argument parser for controlling plotting
import argparse
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
parser.add_argument("--name", type=str, default='../fields/cylinder.xml', help="Name of geometry (default: '../fields/cylinder.xml')")
parser.add_argument("--el", type=int, default=2, help="Number of elements to elevalte the grid (default: 2)")
parser.add_argument("--mp", type=int, default=1, help="Number of patches (default: 0)")
parser.add_argument("--nbpts", type=int, default=100, help="Number of elements used for plot(default: 100)")
args = parser.parse_args()

#==============================================
#==============================================
nbpts    = args.nbpts
RefParm  = args.el
Nump     = args.mp
# geometry = '../fields/quart_annulus.xml'
# geometry = '../fields/circle.xml'
geometry  = '../fields/egg.xml'
#geometry  = args.name

functions = [
    {"name": "density", "expression": '1./(2.+np.cos(10.*np.pi*np.sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2+(z-0.5)**2)))'},
]

ViewGeo(geometry, RefParm, Nump, nbpts = nbpts, functions= functions, plot = args.plot)
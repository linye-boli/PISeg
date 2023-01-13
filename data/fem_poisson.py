import gmsh
import sys 
import numpy as np
import matplotlib.pyplot as plt
import skimage
import os 
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # you may comment out this, I will encounter HDF5 version problem without this line

def cnt_mesh(cnt, model_nm='fg', lc=1e-2):
    gmsh.initialize()
    gmsh.model.add(model_nm)

    # inner 
    for i in range(len(cnt)-1):
        x, y = cnt[i]
        gmsh.model.occ.add_point(x,y,0,lc,i+1)
    
    for i in range(len(cnt)-1):
        if i == len(cnt)-2:
            gmsh.model.occ.add_line(i+1, 1, i+1)
        else:
            gmsh.model.occ.add_line(i+1, i+2, i+1)
    
    curve_fg = (np.arange(len(cnt)-1) + 1).tolist()
    gmsh.model.occ.add_curve_loop(curve_fg, 1)
    gmsh.model.occ.add_plane_surface([1], 1)

    if model_nm == 'bg':
        gmsh.model.occ.add_point(0,0,0,lc,len(cnt) + 1)
        gmsh.model.occ.add_point(0,1,0,lc,len(cnt) + 2)
        gmsh.model.occ.add_point(1,1,0,lc,len(cnt) + 3)
        gmsh.model.occ.add_point(1,0,0,lc,len(cnt) + 4)

        gmsh.model.occ.add_line(len(cnt)+1, len(cnt)+2, len(cnt)+1)
        gmsh.model.occ.add_line(len(cnt)+2, len(cnt)+3, len(cnt)+2)
        gmsh.model.occ.add_line(len(cnt)+3, len(cnt)+4, len(cnt)+3)
        gmsh.model.occ.add_line(len(cnt)+4, len(cnt)+1, len(cnt)+4)

        curve_bg = [len(cnt) + i for i in range(1,5)]
        gmsh.model.occ.add_curve_loop(curve_bg, 2)
        gmsh.model.occ.add_plane_surface([1,2], 2)
    
    gmsh.model.occ.synchronize()

    if model_nm == 'fg':
        gmsh.model.addPhysicalGroup(1, curve_fg, name = "boundary")
        gmsh.model.addPhysicalGroup(2, [1], name = "domain")
    else:
        gmsh.model.addPhysicalGroup(1, curve_fg, name = "boundary")
        gmsh.model.addPhysicalGroup(2, [2], name = "domain")
    
    gmsh.model.mesh.generate(2)
    gmsh.write('./tmp.msh')
    gmsh.finalize()

import meshio
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
import ufl 
from petsc4py.PETSc import ScalarType
from dolfinx.io import gmshio

def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

def msh2xdmf():
    msh = meshio.read("./tmp.msh")
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write("./tmp_mesh.xdmf", triangle_mesh)
    meshio.write("./tmp_mt.xdmf", line_mesh)

def load_xdmf():
    with XDMFFile(MPI.COMM_WORLD, "./tmp_mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        # ct = xdmf.read_meshtags(mesh, name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    with XDMFFile(MPI.COMM_WORLD, "./tmp_mt.xdmf", "r") as xdmf:
        ft = xdmf.read_meshtags(mesh, name="Grid")
    
    return mesh, ft

def solve_poisson(mesh, ft):
    V = fem.FunctionSpace(mesh, ("CG", 1))
    bd_facets = ft.find(1)
    bd_dofs = fem.locate_dofs_topological(V, mesh.topology.dim-1, bd_facets)
    bc = fem.dirichletbc(ScalarType(0), bd_dofs, V)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(mesh, ScalarType(-1))

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh 

def uh_eval(uh, mesh, nx, ny):
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    Z = np.zeros((nx, ny))
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    pts = np.zeros((3,Y.shape[0]))
    pts[0] = Y 
    pts[1] = X

    bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, pts.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, pts.T)

    idx = []
    for i, point in enumerate(pts.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            idx.append(i)
    
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = uh.eval(points_on_proc, cells)

    Z[idx] = u_values.flatten()
    return Z.reshape(nx, ny)

if __name__ == '__main__':
    silhouette = skimage.data.horse()
    cnt = skimage.measure.find_contours(silhouette, 0.5)[0]
    nx, ny = silhouette.shape 
    cnt[:,0] = cnt[:,0]/nx 
    cnt[:,1] = cnt[:,1]/ny

    cnt_mesh(cnt, model_nm='fg')
    msh2xdmf()
    mesh, ft = load_xdmf()
    uh = solve_poisson(mesh, ft) 
    Z = uh_eval(uh, mesh, 128, 128)
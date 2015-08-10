from __future__ import absolute_import, division, print_function

from pymor.domaindescriptions.basic import PolygonalDomain
from pymor.playground.grids.gmsh import GmshBoundaryInfo
from pymor.playground.grids.gmsh import GmshGrid


def discretize_PolygonalDomain(domain_description, mesh_algorithm='meshadapt'):
    assert isinstance(domain_description, PolygonalDomain)
    geo_file_path = domain_description.geo_file_path
    gmsh_file_path = geo_file_path.replace('.geo', '.msh')

    from subprocess import PIPE, Popen
    gmsh = Popen(['gmsh', geo_file_path, '-2', '-algo', mesh_algorithm, '-o', gmsh_file_path], stdout=PIPE)
    gmsh.wait()
    print(gmsh.stdout.read())

    grid = GmshGrid(gmsh_file_path)
    bi = GmshBoundaryInfo(grid)

    return grid, bi

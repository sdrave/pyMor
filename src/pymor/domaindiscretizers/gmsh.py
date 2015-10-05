from __future__ import absolute_import, division, print_function

from pymor.domaindescriptions.basic import PolygonalDomain
from pymor.playground.grids.gmsh import GmshBoundaryInfo
from pymor.playground.grids.gmsh import GmshGrid


def discretize_PolygonalDomain(domain_description=None, geo_file=None, mesh_algorithm='meshadapt', clscale=1.,
                               clmin=0.1, clmax=0.2, options=''):
    assert domain_description is None or geo_file is None
    if domain_description is not None:
        assert isinstance(domain_description, PolygonalDomain)
        geo_file_path = domain_description.geo_file_path
    else:
        geo_file_path = geo_file
    gmsh_file_path = geo_file_path.replace('.geo', '.msh')

    from subprocess import PIPE, Popen
    gmsh = Popen(['gmsh', geo_file_path, '-2', '-algo', mesh_algorithm, '-clscale', str(clscale), '-clmin', str(clmin),
                  '-clmax', str(clmax), options, '-o', gmsh_file_path], stdout=PIPE)
    gmsh.wait()
    print(gmsh.stdout.read())

    grid = GmshGrid(gmsh_file_path)
    bi = GmshBoundaryInfo(grid)

    return grid, bi

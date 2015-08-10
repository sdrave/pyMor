# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.domaindescriptions.interfaces import DomainDescriptionInterface


class RectDomain(DomainDescriptionInterface):
    """Describes a rectangular domain.

    |BoundaryTypes| can be associated edgewise.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain.
    left
        The |BoundaryType| of the left edge.
    right
        The |BoundaryType| of the right edge.
    top
        The |BoundaryType| of the top edge.
    bottom
        The |BoundaryType| of the bottom edge.

    Attributes
    ----------
    domain
    left
    right
    top
    bottom
    """

    def __init__(self, domain=([0, 0], [1, 1]), left=BoundaryType('dirichlet'), right=BoundaryType('dirichlet'),
                 top=BoundaryType('dirichlet'), bottom=BoundaryType('dirichlet')):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        assert left is None or isinstance(left, BoundaryType)
        assert right is None or isinstance(right, BoundaryType)
        assert top is None or isinstance(top, BoundaryType)
        assert bottom is None or isinstance(bottom, BoundaryType)
        self.boundary_types = frozenset({left, right, top, bottom})
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.domain = np.array(domain)

    @property
    def lower_left(self):
        return self.domain[0]

    @property
    def upper_right(self):
        return self.domain[1]

    @property
    def width(self):
        return self.domain[1, 0] - self.domain[0, 0]

    @property
    def height(self):
        return self.domain[1, 1] - self.domain[0, 1]

    @property
    def volume(self):
        return self.width * self.height

    @property
    def diameter(self):
        return np.sqrt(self.width ** 2 + self.height ** 2)

    def __repr__(self):
        left = ', left=' + repr(self.left) if self.left != BoundaryType('dirichlet') else ''
        right = ', right=' + repr(self.right) if self.right != BoundaryType('dirichlet') else ''
        top = ', top=' + repr(self.top) if self.top != BoundaryType('dirichlet') else ''
        bottom = ', bottom=' + repr(self.bottom) if self.bottom != BoundaryType('dirichlet') else ''
        return 'RectDomain({}{})'.format(str(self.domain).replace('\n', ','), left + right + top + bottom)


class CylindricalDomain(DomainDescriptionInterface):
    """Describes a cylindrical domain.

    |BoundaryTypes| can be associated edgewise.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain. The left and right edge are identified.
    top
        The |BoundaryType| of the top edge.
    bottom
        The |BoundaryType| of the bottom edge.

    Attributes
    ----------
    domain
    top
    bottom
    """

    def __init__(self, domain=([0, 0], [1, 1]), top=BoundaryType('dirichlet'), bottom=BoundaryType('dirichlet')):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        assert top is None or isinstance(top, BoundaryType)
        assert bottom is None or isinstance(bottom, BoundaryType)
        self.boundary_types = frozenset({top, bottom})
        self.top = top
        self.bottom = bottom
        self.domain = np.array(domain)

    @property
    def lower_left(self):
        return self.domain[0]

    @property
    def upper_right(self):
        return self.domain[1]

    @property
    def width(self):
        return self.domain[1, 0] - self.domain[0, 0]

    @property
    def height(self):
        return self.domain[1, 1] - self.domain[0, 1]

    @property
    def volume(self):
        return self.width * self.height

    @property
    def diameter(self):
        return np.sqrt(self.width ** 2 + self.height ** 2)

    def __repr__(self):
        top = ', top=' + repr(self.top) if self.top != BoundaryType('dirichlet') else ''
        bottom = ', bottom=' + repr(self.bottom) if self.bottom != BoundaryType('dirichlet') else ''
        return 'CylindricalDomain({}{})'.format(str(self.domain).replace('\n', ','), top + bottom)


class TorusDomain(DomainDescriptionInterface):
    """Describes a domain with the topology of a torus.

    Parameters
    ----------
    domain
        List of two points defining the lower-left and upper-right corner
        of the domain. The left and right edge are identified, as well as the
        bottom and top edge

    Attributes
    ----------
    domain
    """

    def __init__(self, domain=([0, 0], [1, 1])):
        assert domain[0][0] <= domain[1][0]
        assert domain[0][1] <= domain[1][1]
        self.boundary_types = frozenset()
        self.domain = np.array(domain)

    @property
    def lower_left(self):
        return self.domain[0]

    @property
    def upper_right(self):
        return self.domain[1]

    @property
    def width(self):
        return self.domain[1, 0] - self.domain[0, 0]

    @property
    def height(self):
        return self.domain[1, 1] - self.domain[0, 1]

    @property
    def volume(self):
        return self.width * self.height

    @property
    def diameter(self):
        return np.sqrt(self.width ** 2 + self.height ** 2)

    def __repr__(self):
        return 'TorusDomain({})'.format(str(self.domain).replace('\n', ','))


class LineDomain(DomainDescriptionInterface):
    """Describes an interval domain.

    |BoundaryTypes| can be associated edgewise.

    Parameters
    ----------
    domain
        List [x_l, x_r] providing the left and right endpoint.
    left
        The |BoundaryType| of the left endpoint.
    right
        The |BoundaryType| of the right endpoint.

    Attributes
    ----------
    domain
    left
    right
    """

    def __init__(self, domain=(0, 1), left=BoundaryType('dirichlet'), right=BoundaryType('dirichlet')):
        assert domain[0] <= domain[1]
        assert left is None or isinstance(left, BoundaryType)
        assert right is None or isinstance(right, BoundaryType)
        self.boundary_types = frozenset({left, right})
        self.left = left
        self.right = right
        self.domain = np.array(domain)

    @property
    def width(self):
        return self.domain[1] - self.domain[0]

    def __repr__(self):
        left = ', left=' + repr(self.left) if self.left != BoundaryType('dirichlet') else ''
        right = ', right=' + repr(self.right) if self.right != BoundaryType('dirichlet') else ''
        return 'LineDomain({}{})'.format(self.domain, left + right)


class CircleDomain(DomainDescriptionInterface):
    """Describes a domain with the topology of a circle, i.e. a line with
    identified end points.

    Parameters
    ----------
    domain
        List [x_l, x_r] providing the left and right endpoint.

    Attributes
    ----------
    domain
    """

    def __init__(self, domain=(0, 1)):
        assert domain[0] <= domain[1]
        self.domain = np.array(domain)

    @property
    def width(self):
        return self.domain[1] - self.domain[0]

    def __repr__(self):
        return 'CircleDomain({})'.format(self.domain)


class PolygonalDomain(DomainDescriptionInterface):
    point_types = {'Point', 'Physical Point'}
    line_types = {'BSpline', 'Circle', 'CatmullRom', 'Ellipse', 'Line', 'Spline', 'Line Loop', 'Compound Line',
                           'Physical Line'}
    surface_types = {'Plane Surface', 'Ruled Surface', 'Surface Loop', 'Compound Surface', 'Physical Surface'}

    def __init__(self, points, lines, surfaces, boundary_types, geo_file_path):
        self.geo_file_path = geo_file_path
        f = open(geo_file_path, 'w')

        for point_type, ps in points.iteritems():
            assert point_type in self.point_types
            for i, p in enumerate(ps):
                f.write(point_type+'('+str(i+1)+')'+' = '+str(p).replace('[', '{').replace(']', '}')+';\n')

        line_count = 1
        for line_type, ls in lines.iteritems():
            assert line_type in self.line_types
            for l in ls:
                f.write(line_type+'('+str(line_count)+')'+' = '+str(l).replace('[', '{').replace(']', '}')+';\n')
                line_count += 1

        for surface_type, ss in surfaces.iteritems():
            assert surface_type in self.surface_types
            for s in ss:
                f.write(surface_type+'('+str(line_count)+')'+' = '+str(s).replace('[', '{').replace(']', '}')+';\n')
                line_count += 1
        del line_count

        for boundary_type, bs in boundary_types.iteritems():
            for b in bs:
                if str(boundary_type) == 'boundary':
                    f.write('Physical Surface'+'("'+str(boundary_type)+'")'+' = '+str(b).replace('[', '{').replace(']', '}')+';\n')
                else:
                    f.write('Physical Line'+'("'+str(boundary_type)+'")'+' = '+str(b).replace('[', '{').replace(']', '}')+';\n')

        f.close()

    def __repr__(self):
        return 'PolygonalDomain'


class PieDomain(PolygonalDomain):

    def __init__(self, angle, lc=1.):
        self.angle = angle
        from math import pi, cos, sin
        assert (0 < angle) and (angle < 2*pi)

        if angle <= pi:
            points = {'Point': [[0., 0., 0, lc], [1., 0., 0, lc], [cos(angle), sin(angle), 0, lc]]}
            lines = {'Line': [[1, 2], [3, 1]], 'Circle': [[2, 1, 3]], 'Line Loop': [[1, 2, 3]]}
            surfaces = {'Plane Surface': [[4]]}
            boundary_types = {BoundaryType('dirichlet'): [[1, 2]], BoundaryType('neumann'): [[3]], 'boundary': [[5]]}
        else:
            points = {'Point': [[0., 0., 0, lc], [1., 0., 0, lc], [-1., 0., 0, lc], [cos(angle), sin(angle), 0, lc]]}
            lines = {'Line': [[1, 2], [4, 1]], 'Circle': [[2, 1, 3], [3, 1, 4]], 'Line Loop': [[1, 2, 3, 4]]}
            surfaces = {'Plane Surface': [[5]]}
            boundary_types = {BoundaryType('dirichlet'): [[1, 2]], BoundaryType('neumann'): [[3, 4]], 'boundary': [[6]]}
        import os.path
        geo_file_path = os.path.join(os.path.dirname(__file__), '../../../testdata/PieDomain'+str(angle).replace('.', '_')+'.geo')

        super(PieDomain, self).__init__(points, lines, surfaces, boundary_types, geo_file_path)

    def __repr__(self):
        return 'PieDomain({})'.format(self.angle)

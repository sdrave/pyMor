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

    def __init__(self, points, boundary_types, geo_file_path):
        self.geo_file_path = geo_file_path
        f = open(geo_file_path, 'w')

        point_count = 1
        for ps in points:
            for p in ps:
                assert len(p) == 2
                p.extend([0, 0])
                f.write('Point('+str(point_count)+') = '+str(p).replace('[', '{').replace(']', '}')+';\n')
                point_count += 1
        del point_count

        line_count = 1
        elem_count = 1
        line_loop_ids = []
        line_map = {}
        for i, ps in enumerate(points):
            lines = [[line_count+j, line_count+j+1]for j in xrange(len(ps))]
            lines[-1][-1] = lines[0][0]
            line_loop = []
            for l in lines:
                f.write('Line('+str(elem_count)+')'+' = '+str(l).replace('[', '{').replace(']', '}')+';\n')
                line_map[line_count] = elem_count
                line_loop.append(elem_count)
                line_count += 1
                elem_count += 1
            f.write('Line Loop('+str(elem_count)+')'+' = '+str(line_loop).replace('[', '{').replace(']', '}')+';\n')
            line_loop_ids.append(elem_count)
            elem_count += 1
        line_loop_ids.reverse()
        f.write('Plane Surface('+str(elem_count)+')'+' = '+str(line_loop_ids).replace('[', '{').replace(']', '}')+';\n')
        f.write('Physical Surface("boundary") = {'+str(elem_count)+'};\n')
        del line_count
        del elem_count
        del line_loop_ids

        for boundary_type, bs in boundary_types.iteritems():
            f.write('Physical Line'+'("'+str(boundary_type)+'")'+' = '+str([line_map[l_id] for l_id in bs]).replace('[', '{').replace(']', '}')+';\n')
        del line_map

        f.close()

    def __repr__(self):
        return 'PolygonalDomain'


class PieDomain(PolygonalDomain):

    def __init__(self, angle, num_points=100):
        self.angle = angle
        from math import pi, cos, sin
        assert (0 < angle) and (angle < 2*pi)

        points = [[[0, 0]]]
        points[0].extend([[cos(a), sin(a)] for a in np.linspace(start=0, stop=angle, num=num_points, endpoint=True)])
        boundary_types = {BoundaryType('dirichlet'): [1, len(points[0])]}
        boundary_types[BoundaryType('neumann')] = range(2, len(points[0]))

        import os.path
        geo_file_path = os.path.join(os.path.dirname(__file__), '../../../testdata/PieDomain'+str(angle).replace('.', '_')+'.geo')

        super(PieDomain, self).__init__(points, boundary_types, geo_file_path)

    def __repr__(self):
        return 'PieDomain({})'.format(self.angle)

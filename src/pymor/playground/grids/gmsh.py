# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from collections import defaultdict

import numpy as np

from pymor.grids.interfaces import AffineGridInterface
from pymor.grids.referenceelements import triangle

from pymor.domaindescriptions.boundarytypes import BoundaryType
from pymor.grids.interfaces import BoundaryInfoInterface


class GmshParseError(Exception):
    pass


def parse_gmsh_file(f):

    allowed_sections = ['Nodes', 'Elements', 'PhysicalNames', 'Periodic', 'NodeData',
                        'ElementData', 'ElementNodeData']

    supported_sections = ['Nodes', 'Elements', 'PhysicalNames']

    try:

        l = next(f).strip()
        if l != '$MeshFormat':
            raise GmshParseError('expected $MeshFormat, got {}'.format(l))

        l = next(f).strip()
        header = l.split(' ')
        if len(header) != 3:
            raise GmshParseError('header {} has {} fields, expected 3'.format(l, len(header)))

        if header[0] != '2.2':
            raise GmshParseError('wrong file format version: got {}, expected 2.2'.format(header[0]))

        try:
            file_type = int(header[1])
        except ValueError:
            raise GmshParseError('malformed header: expected integer, got {}'.format(header[1]))

        if file_type != 0:
            raise GmshParseError('wrong file type: only ASCII gmsh files are supported')

        try:
            data_size = int(header[2])    # NOQA
        except ValueError:
            raise GmshParseError('malformed header: expected integer, got {}'.format(header[2]))

        l = next(f).strip()
        if l != '$EndMeshFormat':
            raise GmshParseError('expected $EndMeshFormat, got {}'.format(l))

    except StopIteration:
        raise GmshParseError('unexcpected end of file')

    in_section = False
    sections = defaultdict(list)

    for l in f:
        l = l.strip()
        if l == '':
            continue
        if not in_section:
            if not l.startswith('$'):
                raise GmshParseError('expected section name, got {}'.format(l))
            section = l[1:]
            if section not in allowed_sections:
                raise GmshParseError('unknown section type: {}'.format(section))
            if section not in supported_sections:
                raise GmshParseError('unsupported section type: {}'.format(section))
            if section in sections:
                raise GmshParseError('only one {} section allowed'.format(section))
            in_section = True
        elif l.startswith('$'):
            if l != '$End' + section:
                raise GmshParseError('expected $End{}, got {}'.format(section, l))
            in_section = False
        else:
            sections[section].append(l)

    if in_section:
        raise GmshParseError('file ended while in section {}'.format(section))

    # now we parse each section ...

    def parse_nodes(nodes):
        try:
            num_nodes = int(nodes[0])
        except ValueError:
            raise GmshParseError('first line of nodes sections is not a number: {}'.format(nodes[0]))
        if len(nodes) != num_nodes + 1:
            raise GmshParseError('number-of-nodes field does not match number of lines in nodes section')

        nodes = [n.split(' ') for n in nodes[1:]]
        if not all(len(n) == 4 for n in nodes):
            raise GmshParseError('malformed nodes section')

        try:
            nodes = [(int(a), (float(b), float(c), float(d))) for a, b, c, d in nodes]
        except ValueError:
            raise GmshParseError('malformed nodes section')

        return nodes

    def parse_elements(elements):
        try:
            num_elements = int(elements[0])
        except ValueError:
            raise GmshParseError('first line of elements sections is not a number: {}'.format(elements[0]))
        if len(elements) != num_elements + 1:
            raise GmshParseError('number-of-elements field does not match number of lines in elements section')

        elements = [e.split(' ') for e in elements[1:]]
        try:
            elements = [tuple(int(f) for f in e) for e in elements]
        except ValueError:
            raise GmshParseError('malformed elements section')

        element_types = {1: 'line', 2: 'triangle'}
        element_nodes = {'line': 2, 'triangle': 3}

        def parse_line(fields):
            if fields[1] not in element_types:
                raise GmshParseError('element type {} not supported'.format(fields[0]))
            element_type = element_types[fields[1]]
            num_nodes = element_nodes[element_type]
            num_tags = fields[2]
            if len(fields) != num_nodes + num_tags + 3:
                raise GmshParseError('malformed elements section')
            return element_type, (fields[0], tuple(fields[3:3 + num_tags]), fields[3 + num_tags:])

        elements_by_type = defaultdict(list)
        for e in elements:
            t, l = parse_line(e)
            elements_by_type[t].append(l)

        return elements_by_type

    def parse_names(physical_names):
        try:
            num_elements = int(physical_names[0])
        except ValueError:
            raise GmshParseError('first line of physical names sections is not a number: {}'.format(physical_names[0]))
        if len(physical_names) != num_elements + 1:
            raise GmshParseError('number-of-names field does not match number of lines in physical names section')

        physical_names = [pn.split(' ') for pn in physical_names[1:]]
        if not all(len(pn) == 3 for pn in physical_names):
            raise GmshParseError('malformed physical names section')

        try:
            physical_names = [(int(b), int(a), str(c).replace('"', '')) for a, b, c in physical_names]
        except ValueError:
            raise GmshParseError('malformed physical names section')

        return physical_names

    parser_map = {'Nodes': parse_nodes, 'Elements': parse_elements, 'PhysicalNames': parse_names}

    for k, v in sections.iteritems():
        sections[k] = parser_map[k](v)

    return sections


class GmshGrid(AffineGridInterface):

    dim = 2
    dim_outer = 2
    reference_element = triangle

    def __init__(self, gmsh_file):
        self.logger.info('Parsing gmsh file ...')
        self.sections = parse_gmsh_file(gmsh_file)

        self.logger.info('Checking is grid is a 2d triangular grid ...')
        assert {'Nodes', 'Elements', 'PhysicalNames'} <= set(self.sections.keys())
        assert set(self.sections['Elements'].keys()) <= {'line', 'triangle'}
        assert 'triangle' in self.sections['Elements']
        assert all(n[1][2] == 0 for n in self.sections['Nodes'])

        self.logger.info('Creating entity maps ...')
        node_ids = dict(zip([n[0] for n in self.sections['Nodes']], np.arange(len(self.sections['Nodes']), dtype=np.int32)))
        if 'line' in self.sections['Elements']:
            line_ids = dict(zip([l[0] for l in self.sections['Elements']['line']], np.arange(len(self.sections['Elements']['line']), dtype=np.int32)))
        triangle_ids = dict(zip([t[0] for t in self.sections['Elements']['triangle']], np.arange(len(self.sections['Elements']['triangle']), dtype=np.int32)))

        self.logger.info('Building grid topology ...')

        # the lines dict will hold the indices of lines defined by pairs of points
        if 'line' in self.sections['Elements']:
            lines = dict(zip([frozenset(l[2]) for l in self.sections['Elements']['line']], np.arange(len(self.sections['Elements']['line']), dtype=np.int32)))

        nodes_list = [t[2] for t in self.sections['Elements']['triangle']]
        codim2_subentities = np.array([[node_ids[nodes[0]], node_ids[nodes[1]], node_ids[nodes[2]]] for nodes in nodes_list])
        edges_list = [(frozenset(nodes[1:3]), frozenset((nodes[2], nodes[0])), frozenset(nodes[0:2])) for nodes in nodes_list]
        for edges in edges_list:
            lines.update(dict(zip([e for e in edges if e not in lines], np.arange(len(lines), len(lines)+len([e for e in edges if e not in lines]), dtype=np.int32))))
        codim1_subentities = np.array([[lines[edges[0]], lines[edges[1]], lines[edges[2]]] for edges in edges_list])

        del nodes_list
        del edges_list

        self.logger.info('Calculating embeddings ...')
        codim2_centers = np.array([n[1][0:2] for n in self.sections['Nodes']])
        SEC = codim2_centers[codim2_subentities]
        SHIFTS = SEC[:, 0, :]
        TRANS = SEC[:, 1:, :] - SHIFTS[:, np.newaxis, :]
        TRANS = TRANS.swapaxes(1, 2)

        self.__embeddings = (TRANS, SHIFTS)
        self.__subentities = (np.arange(len(codim1_subentities), dtype=np.int32).reshape(-1, 1),
                              codim1_subentities, codim2_subentities)
        self.__sizes = (len(codim1_subentities), len(lines), len(codim2_centers))

    def __str__(self):
        return 'GmshGrid with {} triangles, {} lines, {} vertices'.format(*self.__sizes)

    def size(self, codim=0):
        assert 0 <= codim <= 2, 'Invalid codimension'
        return self.__sizes[codim]

    def subentities(self, codim=0, subentity_codim=None):
        assert 0 <= codim <= 2, 'Invalid codimension'
        if subentity_codim is None:
            subentity_codim = codim + 1
        assert codim <= subentity_codim <= self.dim, 'Invalid subentity codimensoin'
        if codim == 0:
            return self.__subentities[subentity_codim]
        else:
            return super(GmshGrid, self).subentities(codim, subentity_codim)

    def embeddings(self, codim=0):
        if codim == 0:
            return self.__embeddings
        else:
            return super(GmshGrid, self).embeddings(codim)

    @staticmethod
    def test_instances():
        import os.path
        return GmshGrid(open(os.path.join(os.path.dirname(__file__), '../../../../testdata/gmsh_1.msh'))),


class GmshBoundaryInfo(BoundaryInfoInterface):

    def __init__(self, grid):
        self.grid = grid
        sections = grid.sections
        self.boundary_types = [BoundaryType(pn[2]) for pn in sections['PhysicalNames'] if pn[1] == 1]

        name_ids = dict(zip([pn[0] for pn in sections['PhysicalNames']], np.arange(len(sections['PhysicalNames']),
                                                                                   dtype=np.int32)))
        node_ids = dict(zip([n[0] for n in sections['Nodes']], np.arange(len(sections['Nodes']), dtype=np.int32)))
        if 'line' in sections['Elements']:
            line_ids = dict(zip([l[0] for l in sections['Elements']['line']],
                                np.arange(len(sections['Elements']['line']), dtype=np.int32)))

        #masks = {}
        masks2 = {}
        for bt in self.boundary_types:
            #masks[bt] = [np.array([False]*grid.size(1)), np.array([False]*grid.size(2))]
            #masks[bt][0][[line_ids[l[0]] for l in sections['Elements']['line']]] = \
                #[(bt.type == sections['PhysicalNames'][name_ids[l[1][0]]][2]) for l in sections['Elements']['line']]
            #masks[bt][1][np.ravel([[node_ids[n] for n in l[2]] for l in sections['Elements']['line']])] = \
                #np.ravel([(bt.type == sections['PhysicalNames'][name_ids[l[1][0]]][2]) for n in l[2]] for l in
                          #sections['Elements']['line'])
            masks2[bt] = [np.array([False]*grid.size(1)), np.array([False]*grid.size(2))]
            for l in sections['Elements']['line']:
                masks2[bt][0][line_ids[l[0]]] = (bt.type == sections['PhysicalNames'][name_ids[l[1][0]]][2])
                for n in l[2]:
                    if not masks2[bt][1][node_ids[n]]:
                        masks2[bt][1][node_ids[n]] = (bt.type == sections['PhysicalNames'][name_ids[l[1][0]]][2])
        #print(np.all(masks[BoundaryType('dirichlet')][0] == masks2[BoundaryType('dirichlet')][0]))
        #print(np.all(masks[BoundaryType('dirichlet')][1] == masks2[BoundaryType('dirichlet')][1]))
        #print(np.all(masks[BoundaryType('neumann')][0] == masks2[BoundaryType('neumann')][0]))
        #print(np.all(masks[BoundaryType('neumann')][1] == masks2[BoundaryType('neumann')][1]))

        self._masks = masks2

    def mask(self, boundary_type, codim):
        assert 1 <= codim <= self.grid.dim
        assert boundary_type in self.boundary_types
        return self._masks[boundary_type][codim - 1]

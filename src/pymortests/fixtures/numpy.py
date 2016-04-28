# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.playground.vectorarrays.disk import DiskVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray
from pymor.vectorarrays.list import NumpyVector, ListVectorArray

import os; TRAVIS = os.getenv('TRAVIS') == 'true'


def random_integers(count, seed):
    np.random.seed(seed)
    return list(np.random.randint(0, 3200, count))


def numpy_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return NumpyVectorArray(np.random.random((length, dim)), copy=False)


def numpy_list_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return ListVectorArray([NumpyVector(v, copy=False) for v in np.random.random((length, dim))],
                           subtype=(NumpyVector, dim), copy=False)


def numpy_disk_vector_array_factory(length, dim, seed):
    np.random.seed(seed)
    return DiskVectorArray([NumpyVector(v, copy=False) for v in np.random.random((length, dim))],
                           subtype=(NumpyVector, dim))


numpy_vector_array_factory_arguments = \
    zip([0,  0,  1, 43, 102],      # len
        [0, 10, 34, 32,   0],      # dim
        random_integers(5, 123))   # seed

numpy_vector_array_factory_arguments_pairs_with_same_dim = \
    zip([0,  0,  1, 43, 102,  2],         # len1
        [0,  1, 37,  9, 104,  2],         # len2
        [0, 10, 34, 32,   3, 13],         # dim
        random_integers(5, 1234) + [42],  # seed1
        random_integers(5, 1235) + [42])  # seed2

numpy_vector_array_factory_arguments_pairs_with_different_dim = \
    zip([0,  0,  1, 43, 102],      # len1
        [0,  1,  1,  9,  10],      # len2
        [0, 10, 34, 32,   3],      # dim1
        [1, 11,  0, 33,   2],      # dim2
        random_integers(5, 1234),  # seed1
        random_integers(5, 1235))  # seed2


numpy_vector_array_generators = \
    [lambda args=args: numpy_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

numpy_list_vector_array_generators = \
    [lambda args=args: numpy_list_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

numpy_disk_vector_array_generators = [] if TRAVIS else \
    [lambda args=args: numpy_disk_vector_array_factory(*args) for args in numpy_vector_array_factory_arguments]

numpy_vector_array_pair_with_same_dim_generators = \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (numpy_vector_array_factory(l, d, s1),
                                            numpy_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

numpy_list_vector_array_pair_with_same_dim_generators = \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (numpy_list_vector_array_factory(l, d, s1),
                                            numpy_list_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

numpy_disk_vector_array_pair_with_same_dim_generators = [] if TRAVIS else \
    [lambda l=l, l2=l2, d=d, s1=s1, s2=s2: (numpy_disk_vector_array_factory(l, d, s1),
                                            numpy_disk_vector_array_factory(l2, d, s2))
     for l, l2, d, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_same_dim]

numpy_vector_array_pair_with_different_dim_generators = \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (numpy_vector_array_factory(l, d1, s1),
                                                     numpy_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]

numpy_list_vector_array_pair_with_different_dim_generators = \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (numpy_list_vector_array_factory(l, d1, s1),
                                                     numpy_list_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]

numpy_disk_vector_array_pair_with_different_dim_generators = [] if TRAVIS else \
    [lambda l=l, l2=l2, d1=d1, d2=d2, s1=s1, s2=s2: (numpy_disk_vector_array_factory(l, d1, s1),
                                                     numpy_disk_vector_array_factory(l2, d2, s2))
     for l, l2, d1, d2, s1, s2 in numpy_vector_array_factory_arguments_pairs_with_different_dim]

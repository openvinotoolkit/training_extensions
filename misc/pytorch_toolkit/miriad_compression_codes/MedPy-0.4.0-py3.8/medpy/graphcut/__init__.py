"""
====================================================
Graph-cut (max-flow/min-cut) (:mod:`medpy.graphcut`)
====================================================
.. currentmodule:: medpy.graphcut

Provides functionalities to efficiently construct nD graphs from various sources using
arbitrary energy functions (boundary and regional terms). The graph can then be saved in
the Dimacs graph standard [5]_ and/or processed (i.e. cut) using 3rd party graph-cut [1]_
algorithms.

This module makes use of a custom *Boost.Python* [2]_ wrapper written for a modified
version of Boykov and Kolmogorovs max-flow/min-cut algorithm (v3.01) [4]_ that can be found at [3]_. 

Supports voxel- as well as label/region-based graph-cuts.

See below for examples.


Directly generate graphs from image :mod:`medpy.graphcut.generate`
==================================================================
Provides functions to generate graphs efficiently from nD images.
Use together with an energy term from :mod:`~medpy.graphcut.energy_voxel` respectively :mod:`~medpy.graphcut.energy_label`,

.. module:: medpy.graphcut.generate
.. autosummary::
    :toctree: generated/
    
    graph_from_voxels
    graph_from_labels
    
Energy terms for voxel-based graph-cuts :mod:`medpy.graphcut.energy_voxel`
==========================================================================
Run-time optimized energy functions for the graph generation. Voxel based [7]_.

.. module:: medpy.graphcut.energy_voxel
    
Boundary energy terms
---------------------
.. autosummary::
    :toctree: generated/    
    
    boundary_maximum_linear
    boundary_difference_linear
    boundary_maximum_exponential
    boundary_difference_exponential
    boundary_maximum_division
    boundary_difference_division
    boundary_maximum_power
    boundary_difference_power
    
Regional energy terms
---------------------
.. autosummary::
    :toctree: generated/
    
    regional_probability_map
    
Energy terms for label-based graph-cuts :mod:`medpy.graphcut.energy_label`
==========================================================================
Run-time optimized energy functions for the graph generation. Label/Superpixel based [6]_.

.. module:: medpy.graphcut.energy_label

Boundary energy terms
---------------------
.. autosummary::
    :toctree: generated/    
    
    boundary_difference_of_means
    boundary_stawiaski
    boundary_stawiaski_directed
    
Regional energy terms
---------------------
.. autosummary::
    :toctree: generated/
    
    regional_atlas    

Persist a graph :mod:`medpy.graphcut.write`
===========================================
Functions to persist a graph in file formats like Dimacs [5]_, which can be read by external graph-cut algorithms.

.. module:: medpy.graphcut.write
.. autosummary::
    :toctree: generated/
    
    graph_to_dimacs

Graph :mod:`medpy.graphcut.graph`
=================================
Graph objects that can be used to generate a custom graph and execute a graph-cut over it.

.. module:: medpy.graphcut.graph
.. autosummary::
    :toctree: generated/
    
    GCGraph
    Graph
    
Maxflow :mod:`medpy.graphcut.maxflow`
=====================================
C++ wrapper around the max-flow/min-cut implementation of [4]_ using Boost.Python.
Do not use these directly, but rather the graph objects supplied by :mod:`medpy.graphcut.graph`.

.. module:: medpy.graphcut.maxflow
.. autosummary::
    :toctree: generated/
    
    GraphDouble
    GraphFloat
    GraphInt
    
Wrapper :mod:`medpy.graphcut.wrapper`
=====================================
Wrappers for executing graph cuts in a memory-friendly way and other convenience functions.

.. module:: medpy.graphcut.wrapper
.. autosummary::
    :toctree: generated/
    
    split_marker
    graphcut_split
    graphcut_subprocesses
    graphcut_stawiaski

Example of voxel based graph cut
--------------------------------
Import the necessary methods

>>> import numpy
>>> from medpy.io import load, header
>>> from medpy.graphcut import graphcut_from_voxels
>>> from mdepy.graphcut.energy_voxel import boundary_difference_exponential

Loading the images and setting the parameters. Assuming that *image.nii* contains
the image on which to execute the graph-cut, *fgmarkers_image.nii* a binary image
of the same size with True values for the foreground markers and *bgmarkers_image.nii*
respectively for the background markers.

>>> image_data, image_header = load("image.nii")
>>> fgmarkers_image_data, _ = load("fgmarkers_image.nii")
>>> bgmarkers_image_data, _ = load("bgmarkers_image.nii")
>>> sigma = 15.
>>> spacing = header.get_pixel_spacing(image_header)

Building the graph.

>>> gcgraph = graph_from_voxels(fgmarkers_image_data,
                                bgmarkers_image_data,
                                boundary_term = boundary_difference_exponential,
                                boundary_term_args = (image_data, sigma, spacing))

Executing the graph-cut (depending on the image size, this might take a while).

>>> maxflow = gcgraph.maxflow()

Building the resulting segmentation image, with True values for foreground and False
values for background voxels.

>>> result_image_data = numpy.zeros(image_data.size, dtype=numpy.bool)
>>> for idx in range(len(result_image_data)):
        result_image_data[idx] = 0 if gcgraph.termtype.SINK == gcgraph.what_segment(idx) else 1    
>>> result_image_data = result_image_data.reshape(image_data.shape)


References
----------
.. [1] http://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision
.. [2] http://www.boost.org/doc/libs/1_55_0/libs/python/doc/
.. [3] http://vision.csd.uwo.ca/code/
.. [4] Boykov Y., Kolmogorov V. "An Experimental Comparison of Min-Cut/Max-Flow
       Algorithms for Energy Minimization in Vision" In IEEE Transactions on PAMI, Vol. 26,
       No. 9, pp. 1124-1137, Sept. 2004
.. [5] http://lpsolve.sourceforge.net/5.5/DIMACS_maxf.htm
.. [6] Stawiaski J., Decenciere E., Bidlaut F. "Interactive Liver Tumor Segmentation
       Using Graph-cuts and watershed" MICCAI 2008 participation
.. [7] Kolmogorov, Vladimir, and Ramin Zabin. "What energy functions can be minimized
       via graph cuts?." Pattern Analysis and Machine Intelligence, IEEE Transactions
       on 26.2 (2004): 147-159.
"""

# Copyright (C) 2013 Oskar Maier
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# import from compile C++ Python module
from .maxflow import GraphDouble, GraphFloat, GraphInt # this always triggers an error in Eclipse, but is right

# import all functions/methods/classes into the module
from .graph import Graph, GCGraph
from .write import graph_to_dimacs
from .generate import graph_from_labels, graph_from_voxels
from . import energy_label
from . import energy_voxel

# import all sub-modules in the __all__ variable
__all__ = [s for s in dir() if not s.startswith('_')]

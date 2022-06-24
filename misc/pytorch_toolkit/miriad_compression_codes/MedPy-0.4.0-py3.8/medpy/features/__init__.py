"""
=================================================================
Image feature extraction and manipulation (:mod:`medpy.features`)
=================================================================
.. currentmodule:: medpy.features

This package contains various functions for feature extraction and
manipulation in medical images.
 
Intensity :mod:`medpy.features.intensity`
=========================================
Functions to extracts intensity based features. Ready to be
manipulated with :mod:`medpy.features.utilities` and used in
`scikit-learn <http://scikit-learn.org/>`_.

.. module:: medpy.features.intensity
.. autosummary::
    :toctree: generated/
    
    intensities
    centerdistance
    centerdistance_xdminus1
    indices
    shifted_mean_gauss
    mask_distance
    local_mean_gauss
    gaussian_gradient_magnitude
    median
    local_histogram
    hemispheric_difference

Feature representation
----------------------
Features can be one or more dimensional and are kept in the following
structures::

    ===== | == == =====
    s1    | s2 s3 [...]
    f1.1  | 
    f1.2  | 
    f2.1  | 
    f3.1  | 
    f3.2  | 
    [...] | 
    ===== | == == =====

, where each column sX denotes a single sample (voxel) and each row
a features element e.g. f1 is constitutes a 2-dimensional features
and occupies therefore two rows, while f2 is a single element
features with a single row. Entries of this array are of type float.
These feature representation forms are processable by the
`scikit-learn <http://scikit-learn.org/>`_ methods.

Multi-spectral images
---------------------
This package was originally designed for MR images and is therefore
suited to handle multi-spectral data such as RGB and MR images.
Each feature extraction function can be supplied with list/tuple of
images instead of an image. in which case they are considered
co-registered and the feature is extracted from all of them
independently.


Utilities :mod:`medpy.feature.utilities`
========================================
A number of utilities to manipulate feature vectors created with `medpy.features.intensity`.

.. module:: medpy.features.utilities
.. autosummary::
    :toctree: generated/
    
    normalize
    normalize_with_model
    append
    join
    
Histogram :mod:`medy.features.histogram`
========================================
Functions to create various kinds of fuzzy histograms with the fuzzy_histogram function.

.. module:: medpy.features.histogram
.. autosummary::
    :toctree: generated/
    
    fuzzy_histogram
    triangular_membership
    trapezoid_membership
    gaussian_membership
    sigmoidal_difference_membership    

Available membership functions
------------------------------
function (string to pass to `membership` argument of fuzzy_histogram)

* triangular_membership (triangular)
* trapezoid_membership (trapezoid)
* gaussian_membership (gaussian)
* sigmoidal_difference_membership (sigmoid)

The smoothness term
-------------------
The smoothness term determines the affected neighbourhood, e.g., when set to 2, all
values in the range (2 * bin_width + 1/2 bin_wdith) to the left and right of this bin
(center) contribute to this bin. Therefore it determines the smoothing factor of this
fuzzy membership function.
More clearly the smoothness term determines how much the function reaches into the
adjunct bins.

An example of the smoothness parameter::

                  ____________ ________ ____________ ________ ____________
                 /          / \        / \       /  \        / \          \ 
                /          /   \      /   \     /    \      /   \          \ 
               /          /     \    /     \   /      \    /     \          \ 
    ---|----------|----------|----------|----------|----------|----------|----------|----
            x-3        x-2        x-1        x          x+1        x+2        x+3
                  |-nbh      |          |crisp bin |          |      +nbh|

The considered value v is associated with the bin x using crisp (i.e. standard)
histograms. For fuzzy histograms with a smoothness of 2, its membership value for
the bins x-smoothness (x-2) until x+smoothness (x+2) is computed. While it also
might have a membership values for bins further away from x, these are considered to
be only marginal and are therefore nor computed. This leads to a speed-up that is
especially important when a great number of fuzzy histograms have to be computed.

Boundary effect / the guarantee parameter
-----------------------------------------
Values near the left and right border of the histogram might
not contribute with a full value of 1 to the histogram, as part of their contribution
lies outside of the histogram range. To avoid this affect (which can be quite strong
for histograms with few bins and a height smoothness term), set 'guarantee' to True.
The histogram size is then selected to be (left_side - smoothness * bin_width till
right_side + smoothness * bin_width) and therefore neglect all boundary effects.    

Plots of the membership functions can e.g. be found at http://www.atp.ruhr-uni-bochum.de/rt1/syscontrol/node117.html .
    
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

# import all functions/methods/classes into the module
from .histogram import fuzzy_histogram, triangular_membership, trapezoid_membership, \
                       gaussian_membership, sigmoidal_difference_membership
from .intensity import centerdistance, centerdistance_xdminus1, gaussian_gradient_magnitude, \
                       hemispheric_difference, indices, intensities, local_histogram, local_mean_gauss, \
                       median, shifted_mean_gauss, mask_distance
from .utilities import append, join, normalize, normalize_with_model

# import all sub-modules in the __all__ variable
__all__ = [s for s in dir() if not s.startswith('_')]



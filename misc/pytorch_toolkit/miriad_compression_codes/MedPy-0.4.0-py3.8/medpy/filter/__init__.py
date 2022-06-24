"""
===================================================
Image filter and manipulation (:mod:`medpy.filter`)
===================================================
.. currentmodule:: medpy.filter

This package contains various image filters and image
manipulation functions.
 
Smoothing :mod:`medpy.filter.smoothing`
=======================================
Image smoothing / noise reduction in grayscale images.

.. module:: medpy.filter.smoothing
.. autosummary::
    :toctree: generated/
    
    anisotropic_diffusion
    gauss_xminus1d
 
Binary :mod:`medpy.filter.binary`
=================================
Binary image manipulation.

.. module:: medpy.filter.binary
.. autosummary::
    :toctree: generated/
    
    size_threshold
    largest_connected_component
    bounding_box

Image :mod:`medpy.filter.image`
=================================
Grayscale image manipulation.

.. module:: medpy.filter.image
.. autosummary::
    :toctree: generated/
    
    sls
    ssd
    average_filter
    sum_filter
    local_minima
    otsu
    resample
    
Label :mod:`medpy.filter.label`
=================================
Label map manipulation.

.. module:: medpy.filter.label
.. autosummary::
    :toctree: generated/
    
    relabel_map
    relabel
    relabel_non_zero
    fit_labels_to_mask
    
Noise :mod:`medpy.filter.noise`
===============================
Global and local noise estimation in grayscale images.

.. module:: medpy.filter.noise
.. autosummary::
    :toctree: generated/
    
    immerkaer
    immerkaer_local
    separable_convolution
    
    
Utilities :mod:`medpy.filter.utilities`
=======================================
Utilities to apply filters selectively and create your own ones.

.. module:: medpy.filter.utilities
.. autosummary::
    :toctree: generated/
    
    xminus1d
    intersection
    pad
    
Hough transform :mod:`medpy.filter.houghtransform`
==================================================
The hough transform shape detection algorithm.

.. module:: medpy.filter.houghtransform
.. autosummary::
    :toctree: generated/
    
    ght
    ght_alternative
    template_ellipsoid
    template_sphere
    
Intensity range standardization :mod:`medpy.filter.IntensityRangeStandardization`
=================================================================================
A learning method to align the intensity ranges of images.

.. module:: medpy.filter.IntensityRangeStandardization
.. autosummary::
    :toctree: generated/
    
    IntensityRangeStandardization

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

# if __all__ is not set, only the following, explicit import statements are executed
from .binary import largest_connected_component, size_threshold, bounding_box
from .image import sls, ssd, average_filter, sum_filter, otsu, local_minima, resample
from .smoothing import anisotropic_diffusion, gauss_xminus1d
from .label import fit_labels_to_mask, relabel, relabel_map, relabel_non_zero
from .houghtransform import ght, ght_alternative, template_ellipsoid, template_sphere
from .utilities import pad, intersection, xminus1d
from .IntensityRangeStandardization import IntensityRangeStandardization, UntrainedException, InformationLossException, SingleIntensityAccumulationError

# import all sub-modules in the __all__ variable
__all__ = [s for s in dir() if not s.startswith('_')]

"""
=====================================
Metric measures (:mod:`medpy.metric`)
=====================================
.. currentmodule:: medpy.metric

This package provides a number of metric measures that e.g. can be used for testing
and/or evaluation purposes on two binary masks (i.e. measuring their similarity) or
distance between histograms.

Binary metrics (:mod:`medpy.metric.binary`)
===========================================
Metrics to compare binary objects and classification results.

Compare two binary objects
**************************
 
.. module:: medpy.metric.binary

.. autosummary::
    :toctree: generated/
    
    dc
    jc
    hd
    asd
    assd
    precision
    recall
    sensitivity
    specificity
    true_positive_rate
    true_negative_rate
    positive_predictive_value
    ravd
    
Compare two sets of binary objects
**********************************

.. autosummary::
    :toctree: generated/
    
    obj_tpr
    obj_fpr
    obj_asd
    obj_assd
    
Compare to sequences of binary objects
**************************************

.. autosummary::
    :toctree: generated/
    
    volume_correlation
    volume_change_correlation
    
Image metrics (:mod:`medpy.metric.image`)
=========================================
Some more image metrics (e.g. `~medpy.filter.image.sls` and `~medpy.filter.image.ssd`)
can be found in :mod:`medpy.filter.image`. 

.. module:: medpy.metric.image
.. autosummary::
    :toctree: generated/
    
    mutual_information
    
Histogram metrics (:mod:`medpy.metric.histogram`)
=================================================

.. module:: medpy.metric.histogram
.. autosummary::
    :toctree: generated/
    
    chebyshev
    chebyshev_neg
    chi_square
    correlate
    correlate_1
    cosine
    cosine_1
    cosine_2
    cosine_alt
    euclidean
    fidelity_based
    histogram_intersection
    histogram_intersection_1
    jensen_shannon
    kullback_leibler
    manhattan
    minowski
    noelle_1
    noelle_2
    noelle_3
    noelle_4
    noelle_5
    quadratic_forms
    relative_bin_deviation
    relative_deviation

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
from .binary import asd, assd, dc, hd, jc, positive_predictive_value, precision, ravd, recall, sensitivity, specificity, true_negative_rate, true_positive_rate, hd95
from .binary import obj_asd, obj_assd, obj_fpr, obj_tpr
from .binary import volume_change_correlation, volume_correlation
from .histogram import chebyshev, chebyshev_neg, chi_square, correlate, correlate_1, cosine,\
     cosine_1, cosine_2, cosine_alt, euclidean, fidelity_based, histogram_intersection,\
     histogram_intersection_1, jensen_shannon, kullback_leibler, manhattan, minowski, noelle_1,\
     noelle_2, noelle_3, noelle_4, noelle_5, quadratic_forms, relative_bin_deviation, relative_deviation
from .image import mutual_information

# import all sub-modules in the __all__ variable
__all__ = [s for s in dir() if not s.startswith('_')]

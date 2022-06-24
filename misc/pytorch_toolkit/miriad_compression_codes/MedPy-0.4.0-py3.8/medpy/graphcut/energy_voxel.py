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
#
# author Oskar Maier
# version r0.3.0
# since 2012-03-23
# status Release

# build-in modules
import sys

# third-party modules
import numpy
import scipy
import math

# own modules

# code
def regional_probability_map(graph, xxx_todo_changeme):
    r"""
    Regional term based on a probability atlas.
    
    An implementation of a regional term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    Takes an image/graph/map as input where each entry contains a probability value for
    the corresponding GC graph node to belong to the foreground object. The probabilities
    must be in the range :math:`[0, 1]`. The reverse weights are assigned to the sink
    (which corresponds to the background).
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    probability_map : ndarray
        The label image.
    alpha : float
        The energy terms alpha value, balancing between boundary and regional term.
    
    Notes
    -----
    This function requires a probability atlas image of the same shape as the original image
    to be passed along. That means that `~medpy.graphcut.generate.graph_from_labels` has to
    be called with ``regional_term_args`` set to the probability atlas image.
    """
    (probability_map, alpha) = xxx_todo_changeme
    probability_map = scipy.asarray(probability_map)
    probabilities = numpy.vstack([(probability_map * alpha).flat,
                                  ((1 - probability_map) * alpha).flat]).T
    graph.set_tweights_all(probabilities)

def boundary_maximum_linear(graph, xxx_todo_changeme1):
    r"""
    Boundary term processing adjacent voxels maximum value using a linear relationship. 
    
    An implementation of a boundary term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    The same as `boundary_difference_linear`, but working on the gradient image instead
    of the original. See there for details.
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    gradient_image : ndarray
        The gradient image.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the gradient image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    gradient image.
    """
    (gradient_image, spacing) = xxx_todo_changeme1
    gradient_image = scipy.asarray(gradient_image)
    
    # compute maximum intensity to encounter
    max_intensity = float(numpy.abs(gradient_image).max())
    
    def boundary_term_linear(intensities):
        """
        Implementation of a linear boundary term computation over an array.
        """
        # normalize the intensity distances to the interval (0, 1]
        intensities /= max_intensity
        #difference_to_neighbour[difference_to_neighbour > 1] = 1 # this line should not be required, but might be due to rounding errors
        intensities = (1. - intensities) # reverse weights such that high intensity difference lead to small weights and hence more likely to a cut at this edge
        intensities[intensities == 0.] = sys.float_info.min # required to avoid zero values
        return intensities
    
    __skeleton_maximum(graph, gradient_image, boundary_term_linear, spacing)

def boundary_difference_linear(graph, xxx_todo_changeme2):
    r"""
    Boundary term processing adjacent voxels difference value using a linear relationship. 
    
    An implementation of a regional term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    Finds all edges between all neighbours of the image and uses their normalized
    difference in intensity values as edge weight.
    
    The weights are linearly normalized using the maximum possible intensity difference
    of the image. Formally, this value is computed as:
    
    .. math::
    
        \sigma = |max I - \min I|
    
    , where :math:`\min I` constitutes the lowest intensity value in the image, while
    :math:`\max I` constitutes the highest.
    
    The weights between two neighbouring voxels :math:`(p, q)` is then computed as:
    
    .. math::
    
        w(p,q) = 1 - \frac{|I_p - I_q|}{\sigma} + \epsilon
    
    , where :math:`\epsilon` is a infinitively small number and for which
    :math:`w(p, q) \in (0, 1]` holds true.
    
    When the created edge weights should be weighted according to the slice distance,
    provide the list of slice thicknesses via the ``spacing`` parameter. Then all weights
    computed for the corresponding direction are divided by the respective slice
    thickness. Set this parameter to `False` for equally weighted edges.     
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    original_image : ndarray
        The original image.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the original image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    original image.
    """
    (original_image, spacing) = xxx_todo_changeme2
    original_image = scipy.asarray(original_image)
    
    # compute maximum (possible) intensity difference
    max_intensity_difference = float(abs(original_image.max() - original_image.min()))
    
    def boundary_term_linear(intensities):
        """
        Implementation of a linear boundary term computation over an array.
        """
        # normalize the intensity distances to the interval (0, 1]
        intensities /= max_intensity_difference
        #difference_to_neighbour[difference_to_neighbour > 1] = 1 # this line should not be required, but might be due to rounding errors
        intensities = (1. - intensities) # reverse weights such that high intensity difference lead to small weights and hence more likely to a cut at this edge
        intensities[intensities == 0.] = sys.float_info.min # required to avoid zero values
        return intensities
    
    __skeleton_difference(graph, original_image, boundary_term_linear, spacing)

def boundary_maximum_exponential(graph, xxx_todo_changeme3):
    r"""
    Boundary term processing adjacent voxels maximum value using an exponential relationship. 
    
    An implementation of a boundary term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    The same as `boundary_difference_exponential`, but working on the gradient image instead
    of the original. See there for details.
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    gradient_image : ndarray
        The gradient image.
    sigma : float
        The sigma parameter to use in the boundary term.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the gradient image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    gradient image.
    """
    (gradient_image, sigma, spacing) = xxx_todo_changeme3
    gradient_image = scipy.asarray(gradient_image)
    
    def boundary_term_exponential(intensities):
        """
        Implementation of a exponential boundary term computation over an array.
        """
        # apply exp-(x**2/sigma**2)
        intensities = scipy.power(intensities, 2)
        intensities /= math.pow(sigma, 2)
        intensities *= -1
        intensities = scipy.exp(intensities)
        intensities[intensities <= 0] = sys.float_info.min
        return intensities
    
    __skeleton_maximum(graph, gradient_image, boundary_term_exponential, spacing)    

def boundary_difference_exponential(graph, xxx_todo_changeme4):
    r"""
    Boundary term processing adjacent voxels difference value using an exponential relationship.
    
    An implementation of a boundary term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    Finds all edges between all neighbours of the image and uses their difference in
    intensity values as edge weight.
    
    The weights are normalized using an exponential function and a smoothing factor
    :math:`\sigma`. The :math:`\sigma` value has to be supplied manually, since its
    ideal settings differ greatly from application to application.
    
    The weights between two neighbouring voxels :math:`(p, q)` is then computed as
    
    .. math::
    
        w(p,q) = \exp^{-\frac{|I_p - I_q|^2}{\sigma^2}}
    
    , for which :math:`w(p, q) \in (0, 1]` holds true.
    
    When the created edge weights should be weighted according to the slice distance,
    provide the list of slice thicknesses via the ``spacing`` parameter. Then all weights
    computed for the corresponding direction are divided by the respective slice
    thickness. Set this parameter to `False` for equally weighted edges.     
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    original_image : ndarray
        The original image.
    sigma : float
        The sigma parameter to use in the boundary term.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the original image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    original image.
    """
    (original_image, sigma, spacing) = xxx_todo_changeme4
    original_image = scipy.asarray(original_image)
    
    def boundary_term_exponential(intensities):
        """
        Implementation of a exponential boundary term computation over an array.
        """
        # apply exp-(x**2/sigma**2)
        intensities = scipy.power(intensities, 2)
        intensities /= math.pow(sigma, 2)
        intensities *= -1
        intensities = scipy.exp(intensities)
        intensities[intensities <= 0] = sys.float_info.min
        return intensities
    
    __skeleton_difference(graph, original_image, boundary_term_exponential, spacing)
    
def boundary_maximum_division(graph, xxx_todo_changeme5):
    r"""
    Boundary term processing adjacent voxels maximum value using a division relationship. 
    
    An implementation of a boundary term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    The same as `boundary_difference_division`, but working on the gradient image instead
    of the original. See there for details.
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    gradient_image : ndarray
        The gradient image.
    sigma : float
        The sigma parameter to use in the boundary term.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the gradient image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    gradient image.
    """
    (gradient_image, sigma, spacing) = xxx_todo_changeme5
    gradient_image = scipy.asarray(gradient_image)
    
    def boundary_term_division(intensities):
        """
        Implementation of a exponential boundary term computation over an array.
        """
        # apply 1 / (1  + x/sigma)
        intensities /= sigma
        intensities = 1. / (intensities + 1)
        intensities[intensities <= 0] = sys.float_info.min
        return intensities
    
    __skeleton_difference(graph, gradient_image, boundary_term_division, spacing)
    
def boundary_difference_division(graph, xxx_todo_changeme6):
    r"""
    Boundary term processing adjacent voxels difference value using a division relationship. 
    
    An implementation of a boundary term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    Finds all edges between all neighbours of the image and uses their difference in
    intensity values as edge weight.
    
    The weights are normalized using an division function and a smoothing factor
    :math:`\sigma`. The :math:`\sigma` value has to be supplied manually, since its ideal settings
    differ greatly from application to application.
    
    The weights between two neighbouring voxels :math:`(p, q)` is then computed as
    
    .. math::
    
        w(p,q) = \frac{1}{1 + \frac{|I_p - I_q|}{\sigma}}
    
    , for which :math:`w(p, q) \in (0, 1]` holds true.
    
    When the created edge weights should be weighted according to the slice distance,
    provide the list of slice thicknesses via the ``spacing`` parameter. Then all weights
    computed for the corresponding direction are divided by the respective slice
    thickness. Set this parameter to `False` for equally weighted edges.     
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    original_image : ndarray
        The original image.
    sigma : float
        The sigma parameter to use in the boundary term.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the original image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    original image.
    """
    (original_image, sigma, spacing) = xxx_todo_changeme6
    original_image = scipy.asarray(original_image)
    
    def boundary_term_division(intensities):
        """
        Implementation of a division boundary term computation over an array.
        """
        # apply 1 / (1  + x/sigma)
        intensities /= sigma
        intensities = 1. / (intensities + 1)
        intensities[intensities <= 0] = sys.float_info.min
        return intensities
    
    __skeleton_difference(graph, original_image, boundary_term_division, spacing)
    
def boundary_maximum_power(graph, xxx_todo_changeme7):
    """
    Boundary term processing adjacent voxels maximum value using a power relationship. 
    
    An implementation of a boundary term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    The same as `boundary_difference_power`, but working on the gradient image instead
    of the original. See there for details.
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    gradient_image : ndarray
        The gradient image.
    sigma : float
        The sigma parameter to use in the boundary term.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the gradient image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    gradient image.    
    """
    (gradient_image, sigma, spacing) = xxx_todo_changeme7
    gradient_image = scipy.asarray(gradient_image)
    
    def boundary_term_power(intensities):
        """
        Implementation of a power boundary term computation over an array.
        """
        # apply (1 / (1  + x))^sigma
        intensities = 1. / (intensities + 1)
        intensities = scipy.power(intensities, sigma)
        intensities[intensities <= 0] = sys.float_info.min
        return intensities
    
    __skeleton_maximum(graph, gradient_image, boundary_term_power, spacing)       
    
    
def boundary_difference_power(graph, xxx_todo_changeme8):
    r"""
    Boundary term processing adjacent voxels difference value using a power relationship. 
    
    An implementation of a boundary term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_voxels` function.
    
    Finds all edges between all neighbours of the image and uses their difference in
    intensity values as edge weight.
    
    The weights are normalized using an power function and a smoothing factor
    :math:`\sigma`. The :math:`\sigma` value has to be supplied manually, since its
    ideal settings differ greatly from application to application.
    
    The weights between two neighbouring voxels :math:`(p, q)` is then computed as
    
    .. math::
    
        w(p,q) = \frac{1}{1 + |I_p - I_q|}^\sigma
    
    , for which :math:`w(p, q) \in (0, 1]` holds true.
    
    When the created edge weights should be weighted according to the slice distance,
    provide the list of slice thicknesses via the ``spacing`` parameter. Then all weights
    computed for the corresponding direction are divided by the respective slice
    thickness. Set this parameter to `False` for equally weighted edges.     
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    original_image : ndarray
        The original image.
    sigma : float
        The sigma parameter to use in the boundary term.
    spacing : sequence of float or False
        A sequence containing the slice spacing used for weighting the
        computed neighbourhood weight value for different dimensions. If
        `False`, no distance based weighting of the graph edges is performed.
        
    Notes
    -----
    This function requires the original image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_voxels` has to be called with ``boundary_term_args`` set to the
    original image.
    """
    (original_image, sigma, spacing) = xxx_todo_changeme8
    original_image = scipy.asarray(original_image)
    
    def boundary_term_power(intensities):
        """
        Implementation of a exponential boundary term computation over an array.
        """
        # apply (1 / (1  + x))^sigma
        intensities = 1. / (intensities + 1)
        intensities = scipy.power(intensities, sigma)
        intensities[intensities <= 0] = sys.float_info.min
        return intensities
    
    __skeleton_difference(graph, original_image, boundary_term_power, spacing)   

def __skeleton_maximum(graph, image, boundary_term, spacing):
    """
    A skeleton for the calculation of maximum intensity based boundary terms.
    
    This function is equivalent to energy_voxel.__skeleton_difference(), but uses the
    maximum intensity rather than the intensity difference of neighbouring voxels. It is
    therefore suitable to be used with the gradient image, rather than the original
    image.
    
    The computation of the edge weights follows
    
    .. math::
    
        w(p,q) = g(max(I_p, I_q))
    
    ,where :math:`g(\cdot)` is the supplied boundary term function.
    
    @param graph An initialized graph.GCGraph object
    @type graph.GCGraph
    @param image The image to compute on
    @type image numpy.ndarray
    @param boundary_term A function to compute the boundary term over an array of
                         maximum intensities
    @type boundary_term function
    @param spacing A sequence containing the slice spacing used for weighting the
                   computed neighbourhood weight value for different dimensions. If
                   False, no distance based weighting of the graph edges is performed.
    @param spacing sequence | False    
    
    @see energy_voxel.__skeleton_difference() for more details.
    """
    def intensity_maximum(neighbour_one, neighbour_two):
        """
        Takes two voxel arrays constituting neighbours and computes the maximum between
        their intensities.
        """
        return scipy.maximum(neighbour_one, neighbour_two)
        
    __skeleton_base(graph, numpy.abs(image), boundary_term, intensity_maximum, spacing)
    

def __skeleton_difference(graph, image, boundary_term, spacing):
    """
    A skeleton for the calculation of intensity difference based boundary terms.
    
    Iterates over the images dimensions and generates for each an array of absolute
    neighbouring voxel :math:`(p, q)` intensity differences :math:`|I_p, I_q|`. These are
    then passed to the supplied function :math:`g(\cdot)` for for boundary term
    computation. Finally the returned edge weights are added to the graph.
    
    Formally for each edge :math:`(p, q)` of the image, their edge weight is computed as
    
    .. math::
    
        w(p,q) = g(|I_p - I_q|)
    
    ,where :math:`g(\cdot)` is the supplied boundary term function.
    
    The boundary term function has to take an array of intensity differences as only
    parameter and return an array of the same shape containing the edge weights. For the
    implemented function the condition :math:`g(\cdot)\in(0, 1]` must hold true, i.e., it
    has to be strictly positive with :math:`1` as the upper limit.
    
    @note the underlying neighbourhood connectivity is 4 for 2D, 6 for 3D, etc. 
    
    @note This function is able to work with images of arbitrary dimensions, but was only
    tested for 2D and 3D cases.
    
    @param graph An initialized graph.GCGraph object
    @type graph.GCGraph
    @param image The image to compute on
    @type image numpy.ndarray
    @param boundary_term A function to compute the boundary term over an array of
                         absolute intensity differences
    @type boundary_term function
    @param spacing A sequence containing the slice spacing used for weighting the
                   computed neighbourhood weight value for different dimensions. If
                   False, no distance based weighting of the graph edges is performed.
    @param spacing sequence | False    
    """
    def intensity_difference(neighbour_one, neighbour_two):
        """
        Takes two voxel arrays constituting neighbours and computes the absolute
        intensity differences.
        """
        return scipy.absolute(neighbour_one - neighbour_two)
        
    __skeleton_base(graph, image, boundary_term, intensity_difference, spacing)

def __skeleton_base(graph, image, boundary_term, neighbourhood_function, spacing):
    """
    Base of the skeleton for voxel based boundary term calculation.
    
    This function holds the low level procedures shared by nearly all boundary terms.
    
    @param graph An initialized graph.GCGraph object
    @type graph.GCGraph
    @param image The image containing the voxel intensity values
    @type image numpy.ndarray
    @param boundary_term A function to compute the boundary term over an array of
                           absolute intensity differences
    @type boundary_term function
    @param neighbourhood_function A function that takes two arrays of neighbouring pixels
                                  and computes an intensity term from them that is
                                  returned as a single array of the same shape
    @type neighbourhood_function function
    @param spacing A sequence containing the slice spacing used for weighting the
                   computed neighbourhood weight value for different dimensions. If
                   False, no distance based weighting of the graph edges is performed.
    @param spacing sequence | False
    """
    image = scipy.asarray(image)
    image = image.astype(scipy.float_)

    # iterate over the image dimensions and for each create the appropriate edges and compute the associated weights
    for dim in range(image.ndim):
        # construct slice-objects for the current dimension
        slices_exclude_last = [slice(None)] * image.ndim
        slices_exclude_last[dim] = slice(-1)
        slices_exclude_first = [slice(None)] * image.ndim
        slices_exclude_first[dim] = slice(1, None)
        # compute difference between all layers in the current dimensions direction
        neighbourhood_intensity_term = neighbourhood_function(image[slices_exclude_last], image[slices_exclude_first])
        # apply boundary term
        neighbourhood_intensity_term = boundary_term(neighbourhood_intensity_term)
        # compute key offset for relative key difference
        offset_key = [1 if i == dim else 0 for i in range(image.ndim)]
        offset = __flatten_index(offset_key, image.shape)
        # generate index offset function for index dependent offset
        idx_offset_divider = (image.shape[dim] - 1) * offset
        idx_offset = lambda x: int(x / idx_offset_divider) * offset
        
        # weight the computed distanced in dimension dim by the corresponding slice spacing provided
        if spacing: neighbourhood_intensity_term /= spacing[dim]
        
        for key, value in enumerate(neighbourhood_intensity_term.ravel()):
            # apply index dependent offset
            key += idx_offset(key) 
            # add edges and set the weight
            graph.set_nweight(key, key + offset, value, value)    
    
def __flatten_index(pos, shape):
    """
    Takes a three dimensional index (x,y,z) and computes the index required to access the
    same element in the flattened version of the array.
    """
    res = 0
    acc = 1
    for pi, si in zip(reversed(pos), reversed(shape)):
        res += pi * acc
        acc *= si
    return res
    
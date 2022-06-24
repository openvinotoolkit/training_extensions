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
# since 2012-01-18
# status Release

# build-in modules
import math
import sys

# third-party modules
import scipy.ndimage
import numpy

# own modules

# code
def boundary_difference_of_means(graph, label_image, original_image): # label image is not required to hold continuous ids or to start from 1
    r"""
    Boundary term based on the difference of means between adjacent image regions.
    
    An implementation of the boundary term, suitable to be used with the `~medpy.graphcut.generate.graph_from_labels` function.
    
    This simple energy function computes the mean values for all regions. The weights of
    the edges are then determined by the difference in mean values.
    
    The graph weights generated have to be strictly positive and preferably in the
    interval :math:`(0, 1]`. To ensure this, the maximum possible difference in mean values is computed as:
    
    .. math::
    
        \alpha = \|\max \bar{I} - \min \bar{I}\|
    
    , where :math:`\min \bar{I}` constitutes the lowest mean intensity value of all regions in
    the image, while :math:`\max \bar{I}` constitutes the highest mean intensity value With this
    value the weights between a region :math:`x` and its neighbour :math:`y` can be computed:
    
    .. math::
    
        w(x,y) = \max \left( 1 - \frac{\|\bar{I}_x - \bar{I}_y\|}{\alpha}, \epsilon \right)
    
    where :math:`\epsilon` is the smallest floating point step and thus :math:`w(x,y) \in (0, 1]` holds true.
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    label_image : ndarray
        The label image.
    original_image : ndarray
        The original image.
    
    Notes
    -----
    This function requires the original image to be passed along. That means that
    `~medpy.graphcut.generate.graph_from_labels` has to be called with ``boundary_term_args`` set to the
    original image. 
    
    This function is tested on 2D and 3D images and theoretically works for all dimensionalities. 
    """
    # convert to arrays if necessary
    label_image = scipy.asarray(label_image)
    original_image = scipy.asarray(original_image)
    
    if label_image.flags['F_CONTIGUOUS']: # strangely one this one is required to be ctype ordering
        label_image = scipy.ascontiguousarray(label_image)
        
    __check_label_image(label_image)
    
    # create a lookup-table that translates from a label id to its position in the sorted unique vector
    labels_unique = scipy.unique(label_image)
    
    # compute the mean intensities of all regions
    # Note: Bug in mean implementation: means over labels is only computed if the indexes are also supplied
    means = scipy.ndimage.measurements.mean(original_image, labels=label_image, index=labels_unique)
    
    # compute the maximum possible intensity difference
    max_difference = float(abs(min(means) - max(means)))

    # create a lookup table that relates region ids to their respective intensity values
    means = dict(list(zip(labels_unique, means)))

    # get the adjuncancy of the labels
    edges = __compute_edges(label_image)
    
    # compute the difference of means for each adjunct region and add it as a tuple to the dictionary
    if 0. == max_difference: # special case when the divider is zero and therefore all values can be assured to equal zero
        for edge in edges:
            graph.set_nweight(edge[0] - 1, edge[1] - 1, sys.float_info.min, sys.float_info.min)
    else:    
        # compute the difference of means for each adjunct region and add it as a tuple to the dictionary
        for edge in edges:
            value = max(1. - abs(means[edge[0]] - means[edge[1]]) / max_difference, sys.float_info.min)
            graph.set_nweight(edge[0] - 1, edge[1] - 1, value, value)


def boundary_stawiaski(graph, label_image, gradient_image): # label image is not required to hold continuous ids or to start from 1
    r"""
    Boundary term based on the sum of border voxel pairs differences.
     
    An implementation of the boundary term in [1]_, suitable to be used with the `~medpy.graphcut.generate.graph_from_labels` function.
    
    Determines for each two supplied regions the voxels forming their border assuming
    :math:`ndim*2`-connectedness (e.g. :math:`3*2=6` for 3D). From the gradient magnitude values of each
    end-point voxel the border-voxel pairs, the highest one is selected and passed to a
    strictly positive and decreasing function :math:`g(x)`, which is defined as:
    
    .. math::
    
        g(x) = \left(\frac{1}{1+|x|}\right)^k
    
    ,where :math:`k=2`. The final weight :math:`w_{i,j}` between two regions :math:`r_i` and
    :math:`r_j` is then determined by the sum of all these neighbour values:
    
    .. math::
    
        w_{i,j} = \sum_{e_{m,n}\in F_{(r_i,r_j)}}g(\max(|I(m)|,|I(n)|))
    
    , where :math:`F_{(r_i,r_j)}` is the set of border voxel-pairs :math:`e_{m,n}` between
    the regions :math:`r_i` and :math:`r_j` and :math:`|I(p)|` the absolute of the gradient
    magnitude at the voxel :math:`p`
    
    This boundary_function works as an edge indicator in the original image. In simpler
    words the weight (and therefore the energy) is obtained by summing the local contrast
    along the boundaries between two regions.
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    label_image : ndarray
        The label image. Must contain consecutively labelled regions starting from index 1.
    gradient_image : ndarray
        The gradient image.
    
    Notes
    -----
    This function requires the gradient magnitude image of the original image to be passed
    along. That means that `~medpy.graphcut.generate.graph_from_labels` has to be called
    with ``boundary_term_args`` set to the gradient image. This can be obtained e.g. with
    `generic_gradient_magnitude` and `prewitt` from `scipy.ndimage`.
    
    This function is tested on 2D and 3D images and theoretically works for all dimensionalities. 
    
    References
    ----------
    .. [1] Stawiaski J., Decenciere E., Bidlaut F. "Interactive Liver Tumor Segmentation
           Using Graph-cuts and watershed" MICCAI 2008 participation
    """
    # convert to arrays if necessary
    label_image = scipy.asarray(label_image)
    gradient_image = scipy.asarray(gradient_image)
    
    if label_image.flags['F_CONTIGUOUS']: # strangely, this one is required to be ctype ordering
        label_image = scipy.ascontiguousarray(label_image)
        
    __check_label_image(label_image)
        
    for dim in range(label_image.ndim):
        # prepare slicer for all minus last and all minus first "row"
        slicer_from = [slice(None)] * label_image.ndim
        slicer_to = [slice(None)] * label_image.ndim
        slicer_from[dim] = slice(None, -1)
        slicer_to[dim] = slice(1, None)
        # slice views of keys
        keys_from = label_image[slicer_from]
        keys_to = label_image[slicer_to]
        # determine not equal keys
        valid_edges = keys_from != keys_to
        # determine largest gradient
        gradient_max = numpy.maximum(numpy.abs(gradient_image[slicer_from]), numpy.abs(gradient_image[slicer_to]))[valid_edges]
        # determine key order
        keys_max = numpy.maximum(keys_from, keys_to)[valid_edges]
        keys_min = numpy.minimum(keys_from, keys_to)[valid_edges]
        # set edges / nweights
        for k1, k2, val in zip(keys_min, keys_max, gradient_max):
            weight = math.pow(1./(1. + val), 2) # weight contribution of a single pixel
            weight = max(weight, sys.float_info.min)
            graph.set_nweight(k1 - 1 , k2 - 1, weight, weight)


def boundary_stawiaski_directed(graph, label_image, xxx_todo_changeme): # label image is not required to hold continuous ids or to start from 1
    r"""
    Boundary term based on the sum of border voxel pairs differences, directed version.
    
    An implementation of the boundary term in [1]_, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_labels` function.
    
    The basic definition of this term is the same as for `boundary_stawiaski`, but the
    edges of the created graph will be directed.
    
    This boundary_function works as an edge indicator in the original image. In simpler
    words the weight (and therefore the energy) is obtained by summing the local contrast
    along the boundaries between two regions.
    
    When the ``directedness`` parameter is set to zero, the resulting graph will be undirected
    and the behaviour equals `boundary_stawiaski`.
    When it is set to a positive value, light-to-dark transitions are favored i.e. voxels
    with a lower intensity (darker) than the objects tend to be assigned to the object.
    The boundary term is thus changed to:
    
    .. math::
    
          g_{ltd}(x) = \left\{
              \begin{array}{l l}
                g(x) + \beta & \quad \textrm{if $I_i > I_j$}\\
                g(x) & \quad \textrm{if $I_i \leq I_j$}\\
              \end{array} \right.

    With a negative value for ``directedness``, the opposite effect can be achieved i.e.
    voxels with a higher intensity (lighter) than the objects tend to be assigned to the
    object. The boundary term is thus changed to
    
    .. math::
    
      g_{dtl} = \left\{
          \begin{array}{l l}
            g(x) & \quad \textrm{if $I_i > I_j$}\\
            g(x) + \beta & \quad \textrm{if $I_i \leq I_j$}\\
          \end{array} \right.

    Subsequently the :math:`g(x)` in the computation of :math:`w_{i,j}` is substituted by
    :math:`g_{ltd}` resp. :math:`g_{dtl}`. The value :math:`\beta` determines the power of the
    directedness and corresponds to the absolute value of the supplied ``directedness``
    parameter. Experiments showed values between 0.0001 and 0.0003 to be good candidates.
    
    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    label_image : ndarray
        The label image.  Must contain consecutively labelled regions starting from index 1.
    gradient_image : ndarray
        The gradient image.
    directedness : integer
        The weight of the directedness, a positive number to favour
        light-to-dark and a negative to dark-to-light transitions. See function
        description for more details.
    
    Notes
    -----
    This function requires the gradient magnitude image of the original image to be passed
    along. That means that `~medpy.graphcut.generate.graph_from_labels` has to be called
    with ``boundary_term_args`` set to the gradient image. This can be obtained e.g. with
    `generic_gradient_magnitude` and `prewitt` from `scipy.ndimage`.
    
    This function is tested on 2D and 3D images and theoretically works for all dimensionalities.
    
    References
    ----------
    .. [1] Stawiaski J., Decenciere E., Bidlaut F. "Interactive Liver Tumor Segmentation
           Using Graph-cuts and watershed" MICCAI 2008 participation    
    """
    (gradient_image, directedness) = xxx_todo_changeme
    label_image = scipy.asarray(label_image)
    gradient_image = scipy.asarray(gradient_image)
    
    if label_image.flags['F_CONTIGUOUS']: # strangely one this one is required to be ctype ordering
        label_image = scipy.ascontiguousarray(label_image)
        
    __check_label_image(label_image)
        
    beta = abs(directedness)
        
    def addition_directed_ltd(key1, key2, v1, v2, dic): # for light-to-dark # tested
        "Takes a key defined by two uints, two voxel intensities and a dict to which it adds g(v1, v2)."
        if not key1 == key2: # do not process voxel pairs which belong to the same region
            # The function used to compute the weight contribution of each voxel pair
            weight = math.pow(1./(1. + max(abs(v1), abs(v2))), 2)
            # ensure that no value is zero; this can occur due to rounding errors
            weight = max(weight, sys.float_info.min)
            # add weighted values to already existing edge
            if v1 > v2: graph.set_nweight(key1 - 1, key2 - 1, min(1, weight + beta), weight)
            else: graph.set_nweight(key1 - 1, key2 - 1, weight, min(1, weight + beta))
            
    def addition_directed_dtl(key1, key2, v1, v2): # for dark-to-light # tested
        "Takes a key defined by two uints, two voxel intensities and a dict to which it adds g(v1, v2)."
        if not key1 == key2: # do not process voxel pairs which belong to the same region
            # The function used to compute the weight contribution of each voxel pair
            weight = math.pow(1./(1. + max(abs(v1), abs(v2))), 2)
            # ensure that no value is zero; this can occur due to rounding errors
            weight = max(weight, sys.float_info.min)
            # add weighted values to already existing edge
            if v1 > v2: graph.set_nweight(key1 - 1, key2 - 1, weight, min(1, weight + beta))
            else: graph.set_nweight(key1 - 1, key2 - 1, min(1, weight + beta), weight)
                                                  
    # pick and vectorize the function to achieve a speedup
    if 0 > directedness:
        vaddition = scipy.vectorize(addition_directed_dtl)
    else:
        vaddition = scipy.vectorize(addition_directed_ltd)
    
    # iterate over each dimension
    for dim in range(label_image.ndim):
        slices_x = []
        slices_y = []
        for di in range(label_image.ndim):
            slices_x.append(slice(None, -1 if di == dim else None))
            slices_y.append(slice(1 if di == dim else None, None))
        vaddition(label_image[slices_x],
                  label_image[slices_y],
                  gradient_image[slices_x],
                  gradient_image[slices_y])

def regional_atlas(graph, label_image, xxx_todo_changeme1): # label image is required to hold continuous ids starting from 1
    r"""
    Regional term based on a probability atlas.
    
    An implementation of a regional term, suitable to be used with the
    `~medpy.graphcut.generate.graph_from_labels` function.
    
    This regional term introduces statistical probability of a voxel to belong to the
    object to segment. It computes the sum of all statistical atlas voxels under each
    region and uses this value as terminal node weight for the graph cut.

    Parameters
    ----------
    graph : GCGraph
        The graph to add the weights to.
    label_image : ndarray
        The label image.
    probability_map : ndarray
        The probability atlas image associated with the object to segment.
    alpha : float
        The energy terms alpha value, balancing between boundary and regional term. 
    
    Notes
    -----
    This function requires a probability atlas image of the same shape as the original image
    to be passed along. That means that `~medpy.graphcut.generate.graph_from_labels` has to
    be called with ``regional_term_args`` set to the probability atlas image.
    
    This function is tested on 2D and 3D images and theoretically works for all dimensionalities.    
    """
    (probability_map, alpha) = xxx_todo_changeme1
    label_image = scipy.asarray(label_image)
    probability_map = scipy.asarray(probability_map)
    __check_label_image(label_image)
    
    # finding the objects in the label image (bounding boxes around regions)
    objects = scipy.ndimage.find_objects(label_image)
    
    # iterate over regions and compute the respective sums of atlas values
    for rid in range(1, len(objects) + 1):
        weight = scipy.sum(probability_map[objects[rid - 1]][label_image[objects[rid - 1]] == rid])
        graph.set_tweight(rid - 1, alpha * weight, -1. * alpha * weight) # !TODO: rid's inside the graph start from 0 or 1? => seems to start from 0
        # !TODO: I can exclude source and sink nodes from this!
        # !TODO: I only have to do this in the range of the atlas objects!


def __compute_edges(label_image):
    """
    Computes the region neighbourhood defined by a star shaped n-dimensional structuring
    element (as returned by scipy.ndimage.generate_binary_structure(ndim, 1)) for the
    supplied region/label image.
    Note The returned set contains neither duplicates, nor self-references
    (i.e. (id_1, id_1)), nor reversed references (e.g. (id_1, id_2) and (id_2, id_1).
    
    @param label_image An image with labeled regions (nD).
    @param return A set with tuples denoting the edge neighbourhood.
    """
    return __compute_edges_nd(label_image)
    
def __compute_edges_nd(label_image):
    """
    Computes the region neighbourhood defined by a star shaped n-dimensional structuring
    element (as returned by scipy.ndimage.generate_binary_structure(ndim, 1)) for the
    supplied region/label image.
    Note The returned set contains neither duplicates, nor self-references
    (i.e. (id_1, id_1)), nor reversed references (e.g. (id_1, id_2) and (id_2, id_1).
    
    @param label_image An image with labeled regions (nD).
    @param return A set with tuples denoting the edge neighbourhood.
    """
    Er = set()
   
    def append(v1, v2):
        if v1 != v2:
            Er.update([(min(v1, v2), max(v1, v2))])
        
    vappend = scipy.vectorize(append)
   
    for dim in range(label_image.ndim):
        slices_x = []
        slices_y = []
        for di in range(label_image.ndim):
            slices_x.append(slice(None, -1 if di == dim else None))
            slices_y.append(slice(1 if di == dim else None, None))
        vappend(label_image[slices_x], label_image[slices_y])
        
    return Er

def __check_label_image(label_image):
    """Check the label image for consistent labelling starting from 1."""
    encountered_indices = scipy.unique(label_image)
    expected_indices = scipy.arange(1, label_image.max() + 1)
    if not encountered_indices.size == expected_indices.size or \
       not (encountered_indices == expected_indices).all():
        raise AttributeError('The supplied label image does either not contain any regions or they are not labeled consecutively starting from 1.')

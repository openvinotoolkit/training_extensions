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
# version r0.2.1
# since 2013-10-14
# status Release

# build-in modules
from operator import  lt, le, gt, ge, ne, eq

# third-party modules
import numpy 
from scipy.ndimage.measurements import label

# own modules

# code
def size_threshold(img, thr, comp='lt', structure = None):
    r"""
    Removes binary objects from an image identified by a size threshold.
    
    The unconnected binary objects in an image are identified and all removed
    whose size compares (e.g. less-than) to a supplied threshold value.
    
    The threshold ``thr`` can be any positive integer value. The comparison operator
    can be one of lt, le, gt, ge, ne or eq. The operators used are the functions of
    the same name supplied by the `operator` module of python.
    
    Parameters
    ----------
    img : array_like
        An array containing connected objects. Will be cast to type numpy.bool.
    thr : int
        Integer defining the threshold size of the binary objects to remove.
    comp : {'lt', 'le', 'gt', 'ge', 'ne', 'eq'}
        The type of comparison to perform. Use e.g. 'lt' for less-than.
    structure : array of ints, optional
        A structuring element that defines feature connections.
        ``structure`` must be symmetric. If no structuring element is provided,
        one is automatically generated with a squared connectivity equal to
        one. That is, for a 2-D ``input`` array, the default structuring element
        is::
        
            [[0,1,0],
             [1,1,1],
             [0,1,0]]
    
    Returns
    -------
    binary_image : ndarray
        The supplied binary image with all objects removed that positively compare
        to the threshold ``thr`` using the comparison operator defined with ``comp``.
        
    Notes
    -----
    If your voxel size is no isotrop i.e. of side-length 1 for all dimensions, simply
    divide the supplied threshold through the real voxel size.
    """
    
    operators = {'lt': lt, 'le': le, 'gt': gt, 'ge': ge, 'eq': eq, 'ne': ne}
    
    img = numpy.asarray(img).astype(numpy.bool)
    if comp not in operators:
        raise ValueError("comp must be one of {}".format(list(operators.keys())))
    comp = operators[comp]
    
    labeled_array, num_features = label(img, structure)
    for oidx in range(1, num_features + 1):
        omask = labeled_array == oidx
        if comp(numpy.count_nonzero(omask), thr):
            img[omask] = False
            
    return img

def largest_connected_component(img, structure = None):
    r"""
    Select the largest connected binary component in an image.
    
    Treats all zero values in the input image as background and all others as foreground.
    The return value is an binary array of equal dimensions as the input array with TRUE
    values where the largest connected component is situated.
    
    Parameters
    ----------
    img : array_like
        An array containing connected objects. Will be cast to type numpy.bool.
    structure : array_like
        A structuring element that defines the connectivity. Structure must be symmetric.
        If no structuring element is provided, one is automatically generated with a
        squared connectivity equal to one.
    
    Returns
    -------
    binary_image : ndarray
        The supplied binary image with only the largest connected component remaining.
    """   
    labeled_array, num_features = label(img, structure)
    component_sizes = [numpy.count_nonzero(labeled_array == label_idx) for label_idx in range(1, num_features + 1)]
    largest_component_idx = numpy.argmax(component_sizes) + 1

    out = numpy.zeros(img.shape, numpy.bool)  
    out[labeled_array == largest_component_idx] = True
    return out

def bounding_box(img):
    r"""
    Return the bounding box incorporating all non-zero values in the image.
    
    Parameters
    ----------
    img : array_like
        An array containing non-zero objects.
        
    Returns
    -------
    bbox : a list of slicer objects defining the bounding box
    """
    locations = numpy.argwhere(img)
    mins = locations.min(0)
    maxs = locations.max(0) + 1
    return [slice(x, y) for x, y in zip(mins, maxs)]

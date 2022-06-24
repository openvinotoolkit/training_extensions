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
# version r0.1.1
# since 2012-02-07
# status Development

# build-in modules

# third-party modules
import scipy

# own modules
from ..core.exceptions import ArgumentError

# code
def relabel_map(label_image, mapping, key=lambda x, y: x[y]):
    r"""
    Relabel an image using the supplied mapping.
    
    The ``mapping`` can be any kind of subscriptable object. The respective region id is used
    to access the new value from the ``mapping``. The ``key`` keyword parameter can be used to
    supply another access function. The ``key`` function must have the signature
    key(mapping, region-id) and return the new region-id to assign.
    
    Parameters
    ----------
    label_image : array_like
        A nD label map.
    mapping : dictionary or subscriptable object
        A mapping object.
    key : function
        Can be used to defined the key-access to the ``mapping`` object.
    
    Returns
    -------
    relabel_map : ndarray
        A label map with new region ids.
    
    Raises
    ------
    ArgumentError
        If a region id is missing in the supplied mapping
    """    
    label_image = scipy.array(label_image)
    
    def _map(x):
        try:
            return key(mapping, x)
        except Exception as e:
            raise ArgumentError('No conversion for region id {} found in the supplied mapping. Error: {}'.format(x, e))
    
    vmap = scipy.vectorize(_map, otypes=[label_image.dtype])
         
    return vmap(label_image)

def relabel(label_image, start = 1):
    r"""
    Relabel the regions of a label image.
    Re-processes the labels to make them consecutively and starting from start.
    
    Parameters
    ----------
    label_image : array_like
        A nD label map.
    start : integer
        The id of the first label to assign
    
    Returns
    -------
    relabel_map : ndarray
        The relabelled label map.
        
    See also
    --------
    relabel_non_zero
    """
    label_image = scipy.asarray(label_image)
    mapping = {}
    rav = label_image.ravel()
    for i in range(len(rav)):
        if not rav[i] in mapping:
            mapping[rav[i]] = start
            start += 1
        rav[i] = mapping[rav[i]]
    return rav.reshape(label_image.shape)

def relabel_non_zero(label_image, start = 1):
    r""" 
    Relabel the regions of a label image.
    Re-processes the labels to make them consecutively and starting from start.
    Keeps all zero (0) labels, as they are considered background.
    
    Parameters
    ----------
    label_image : array_like
        A nD label map.
    start : integer
        The id of the first label to assign
    
    Returns
    -------
    relabel_map : ndarray
        The relabelled label map.
        
    See also
    --------
    relabel  
    """
    if start <= 0: raise ArgumentError('The starting value can not be 0 or lower.')
    
    l = list(scipy.unique(label_image))
    if 0 in l: l.remove(0)
    mapping = dict()
    mapping[0] = 0
    for key, item in zip(l, list(range(start, len(l) + start))):
        mapping[key] = item
    
    return relabel_map(label_image, mapping)


def fit_labels_to_mask(label_image, mask):
    r"""
    Reduces a label images by overlaying it with a binary mask and assign the labels
    either to the mask or to the background. The resulting binary mask is the nearest
    expression the label image can form of the supplied binary mask.
    
    Parameters
    ----------
    label_image : array_like
        A nD label map.
    mask : array_like
        A mask image, i.e., a binary image with False for background and True for foreground.
        
    Returns
    -------
    best_fit : ndarray
        The best fit of the labels to the mask.
    
    Raises
    ------         
    ValueError
        If ``label_image`` and ``mask`` are not of the same shape.
    """
    label_image = scipy.asarray(label_image)
    mask = scipy.asarray(mask, dtype=scipy.bool_)

    if label_image.shape != mask.shape:
        raise ValueError('The input images must be of the same shape.')
    
    # prepare collection dictionaries
    labels = scipy.unique(label_image)
    collection = {}
    for label in labels:
        collection[label] = [0, 0, []]  # size, union, points
    
    # iterate over the label images pixels and collect position, size and union
    for x in range(label_image.shape[0]):
        for y in range(label_image.shape[1]):
            for z in range(label_image.shape[2]):
                entry = collection[label_image[x,y,z]]
                entry[0] += 1
                if mask[x,y,z]: entry[1] += 1
                entry[2].append((x,y,z))
                
    # select labels that are more than half in the mask
    for label in labels:
        if collection[label][0] / 2. >= collection[label][1]:
            del collection[label]
                
    # image_result = numpy.zeros_like(mask) this is eq. to mask.copy().fill(0), which directly applied does not allow access to the rows and colums: Why?
    image_result = mask.copy()
    image_result.fill(False)         

    # add labels to result mask
    for label, data in list(collection.items()):
        for point in data[2]:
            image_result[point] = True
            
    return image_result

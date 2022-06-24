#!/usr/bin/python3

"""
Re-samples a binary image according to a supplied voxel spacing.

Copyright (C) 2013 Oskar Maier

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# build-in modules
import os
import logging
import argparse

# third-party modules
import numpy
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion
from scipy.ndimage.measurements import label

# own modules
from medpy.core import Logger
from medpy.filter import resample, bounding_box
from medpy.utilities import argparseu
from medpy.io import load, save, header

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2014-11-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Re-samples a binary image according to a supplied voxel spacing.

For an optimal results without outliers or holes in the case of up-sampling, the required
number of additional slices is added using shape based interpolation. All down-sampling
and the remaining small up-sampling operations are then executed with a nearest
neighbourhood interpolation of a chosen order.

BSpline is used for interpolation. A order between 0 and 5 can be selected. The default
is 0 (= nearest neighbour). In some rare case an order of 1 (= linear) might be
necessary.

Note that the pixel data type of the input image is treated as binary.

Copyright (C) 2013 Oskar Maier
This program comes with ABSOLUTELY NO WARRANTY; This is free software,
and you are welcome to redistribute it under certain conditions; see
the LICENSE file or <http://www.gnu.org/licenses/> for details.   
"""

# code
def main():
    parser = getParser()
    args = getArguments(parser)

    # prepare logger
    logger = Logger.getInstance()
    if args.debug: logger.setLevel(logging.DEBUG)
    elif args.verbose: logger.setLevel(logging.INFO)
    
    # loading input images
    img, hdr = load(args.input)
    img = img.astype(numpy.bool)
    
    # check spacing values
    if not len(args.spacing) == img.ndim:
        parser.error('The image has {} dimensions, but {} spacing parameters have been supplied.'.format(img.ndim, len(args.spacing)))
        
    # check if output image exists
    if not args.force:
        if os.path.exists(args.output):
            parser.error('The output image {} already exists.'.format(args.output)) 
        
    logger.debug('target voxel spacing: {}'.format(args.spacing))
    
    # determine number of required complete slices for up-sampling
    vs = header.get_pixel_spacing(hdr)
    rcss = [int(y // x - 1) for x, y in zip(args.spacing, vs)] # TODO: For option b, remove the - 1; better: no option b, since I am rounding later anyway
    
    # remove negatives and round up to next even number
    rcss = [x if x > 0 else 0 for x in rcss]
    rcss = [x if 0 == x % 2 else x + 1 for x in rcss]
    logger.debug('intermediate slices to add per dimension: {}'.format(rcss))
    
    # for each dimension requiring up-sampling, from the highest down, perform shape based slice interpolation
    logger.info('Adding required slices using shape based interpolation.')
    for dim, rcs in enumerate(rcss):
        if rcs > 0:
            logger.debug('adding {} intermediate slices to dimension {}'.format(rcs, dim))
            img = shape_based_slice_interpolation(img, dim, rcs)
            logger.debug('resulting new image shape: {}'.format(img.shape))
            
    # compute and set new voxel spacing
    nvs = [x / (y + 1.) for x, y in zip(vs, rcss)]
    header.set_pixel_spacing(hdr, nvs)
    logger.debug('intermediate voxel spacing: {}'.format(nvs))
    
    # interpolate with nearest neighbour
    logger.info('Re-sampling the image with a b-spline order of {}.'.format(args.order))
    img, hdr = resample(img, hdr, args.spacing, args.order, mode='nearest')
    
    # saving the resulting image
    save(img, args.output, hdr, args.force)
    
def shape_based_slice_interpolation(img, dim, nslices):
    """
    Adds `nslices` slices between all slices of the binary image `img` along dimension
    `dim` respecting the original slice values to be situated in the middle of each
    slice. Extrapolation situations are handled by simple repeating.
    
    Interpolation of new slices is performed using shape based interpolation.
    
    Parameters
    ----------
    img : array_like
        A n-dimensional image.
    dim : int
        The dimension along which to add slices.
    nslices : int
        The number of slices to add. Must be an even number.
    
    Returns
    -------
    out : ndarray
        The re-sampled image.
    """
    # check arguments
    if not 0 == nslices % 2:
        raise ValueError('nslices must be an even number')
    
    out = None
    slicer = [slice(None)] * img.ndim
    chunk_full_shape = list(img.shape)
    chunk_full_shape[dim] = nslices + 2
     
    for sl1, sl2 in zip(numpy.rollaxis(img, dim)[:-1], numpy.rollaxis(img, dim)[1:]):
        if 0 == numpy.count_nonzero(sl1) and 0 == numpy.count_nonzero(sl2):
            chunk = numpy.zeros(chunk_full_shape, dtype=numpy.bool)
        else:
            chunk = shape_based_slice_insertation_object_wise(sl1, sl2, dim, nslices)
        if out is None:
            out = numpy.delete(chunk, -1, dim)
        else:
            out = numpy.concatenate((out, numpy.delete(chunk, -1, dim)), dim)
    
    slicer[dim] = numpy.newaxis    
    out = numpy.concatenate((out, sl2[slicer]), dim)
    
    slicer[dim] = slice(0, 1)
    for _ in range(nslices // 2):
        out = numpy.concatenate((img[slicer], out), dim)
    slicer[dim] = slice(-1, None)
    for _ in range(nslices // 2):
        out = numpy.concatenate((out, img[slicer]), dim)
        
    return out

def shape_based_slice_insertation_object_wise(sl1, sl2, dim, nslices, order=3):
    """
    Wrapper to apply `shape_based_slice_insertation()` for each binary object
    separately to ensure correct extrapolation behaviour.
    """
    out = None
    sandwich = numpy.concatenate((sl1[numpy.newaxis], sl2[numpy.newaxis]), 0)
    label_image, n_labels = label(sandwich)
    for lid in range(1, n_labels + 1):
        _sl1, _sl2 = label_image == lid
        _out = shape_based_slice_insertation(_sl1, _sl2, dim, nslices, order=3)
        if out is None:
            out = _out
        else:
            out |= _out
    return out
    
def shape_based_slice_insertation(sl1, sl2, dim, nslices, order=3):
    """
    Insert `nslices` new slices between `sl1` and `sl2` along dimension `dim` using shape
    based binary interpolation.
    
    Extrapolation is handled adding `nslices`/2 step-wise eroded copies of the last slice
    in each direction.
    
    Parameters
    ----------
    sl1 : array_like
        First slice. Treated as binary data.
    sl2 : array_like
        Second slice. Treated as binary data.
    dim : int
        The new dimension along which to add the new slices.
    nslices : int
        The number of slices to add.
    order : int
        The b-spline interpolation order for re-sampling the distance maps.
        
    Returns
    -------
    out : ndarray
        A binary image of size `sl1`.shape() extend by `nslices`+2 along the new
        dimension `dim`. The border slices are the original slices `sl1` and `sl2`.
    """
    sl1 = sl1.astype(numpy.bool)
    sl2 = sl2.astype(numpy.bool)
    
    # extrapolation through erosion
    if 0 == numpy.count_nonzero(sl1):
        slices = [sl1]
        for _ in range(nslices / 2):
            slices.append(numpy.zeros_like(sl1))
        for i in range(1, nslices / 2 + nslices % 2 + 1)[::-1]:
            slices.append(binary_erosion(sl2, iterations=i))
        slices.append(sl2)
        return numpy.rollaxis(numpy.asarray(slices), 0, dim + 1)
        #return numpy.asarray([sl.T for sl in slices]).T
    elif 0 ==numpy.count_nonzero(sl2):
        slices = [sl1]
        for i in range(1, nslices / 2 + 1):
            slices.append(binary_erosion(sl1, iterations=i))
        for _ in range(0, nslices / 2 + nslices % 2):
            slices.append(numpy.zeros_like(sl2))
        slices.append(sl2)
        return numpy.rollaxis(numpy.asarray(slices), 0, dim + 1)
        #return numpy.asarray([sl.T for sl in slices]).T
    
    # interpolation shape based 
    # note: distance_transform_edt shows strange behaviour for ones-arrays
    dt1 = distance_transform_edt(~sl1) - distance_transform_edt(sl1)
    dt2 = distance_transform_edt(~sl2) - distance_transform_edt(sl2)
    
    slicer = [slice(None)] * dt1.ndim
    slicer = slicer[:dim] + [numpy.newaxis] + slicer[dim:]
    out = numpy.concatenate((dt1[slicer], dt2[slicer]), axis=dim)
    zoom_factors = [1] * dt1.ndim
    zoom_factors = zoom_factors[:dim] + [(nslices + 2)/2.] + zoom_factors[dim:]
    out = zoom(out, zoom_factors, order=order)
    
    return out <= 0
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    if args.order < 0 or args.order > 5:
        parser.error('The order has to be a number between 0 and 5.')   
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__description__)
    parser.add_argument('input', help='the input image')
    parser.add_argument('output', help='the output image')
    parser.add_argument('spacing', type=argparseu.sequenceOfFloatsGt, help='the desired voxel spacing in colon-separated values, e.g. 1.2,1.2,5.0')
    parser.add_argument('-o', '--order', type=int, default=0, dest='order', help='the bspline order, default is 0 (= nearest neighbour)')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='overwrite existing files')
    return parser
    
if __name__ == "__main__":
    main()    

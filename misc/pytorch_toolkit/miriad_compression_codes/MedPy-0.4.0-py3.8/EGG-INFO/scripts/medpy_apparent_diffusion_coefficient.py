#!/usr/bin/python3

"""
Computes the apparent diffusion coefficient from two diffusion weighted MRI images.

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
import argparse
import logging

# third-party modules
import numpy
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation,\
    binary_erosion

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save, header
from medpy.filter import otsu
from medpy.core.exceptions import ArgumentError
from medpy.filter.binary import largest_connected_component


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2013-07-18"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Computes the apparent diffusion coefficient from two diffusion weighted
MRI images. The output image will be of type float.

Normally diffusion weight (DW) MRI images are acquired once with a
b-value of 0 (which we call b0) and once with another b-value (which we
call bx) such as 500, 800 or 1000. The latter is typical for brain MRIs.
This results in a single b0 DW image and three bx DW images, one for each
direction.

Usually the three bx DW images are already combined into an isotropic
average image (which we call abx) denoting the length of the three-dimensional
vector formed by the the three bx images.

The formula presented in [1] is applied to the b0 and abx images to
compute the apparent diffusion coefficient (ADC):

ADC = -bx-value * ln(abx-image / b0-image)

To cope with zero-values in the images, we apply a-priori a
thresholding to the b0 + abx DW image, set all lower values to 0 and
apply the formula only to the remaining intensities. Note that the
default threshold is chosen using Otsu's and is good for most cases.
(Thanks to Nils at the UKE in Hamburg, Germany for this hint!)

We restrain from implementing a method working on more DW images, that
were acquired with multiple b-values, as [2] observed that this might
lead to worse results.

[1] "Understanding Diffusion MR Imaging Techniques: From Scalar
Diffusion-weighted Imaging to Diffusion Tensor Imaging and Beyond" by
Patric Hagmann et al.
[2] "Understanding the Mathematics Involved in Calculating Apparent
Diffusion Coefficient Maps" by Michael Yong Park and Jae Young Byun

Copyright (C) 2013 Oskar Maier
This program comes with ABSOLUTELY NO WARRANTY; This is free software,
and you are welcome to redistribute it under certain conditions; see
the LICENSE file or <http://www.gnu.org/licenses/> for details.   
                  """

# code
def main():
    args = getArguments(getParser())

    # prepare logger
    logger = Logger.getInstance()
    if args.debug: logger.setLevel(logging.DEBUG)
    elif args.verbose: logger.setLevel(logging.INFO)
    
    # loading input images
    b0img, b0hdr = load(args.b0image)
    bximg, bxhdr = load(args.bximage)
    
    # convert to float
    b0img = b0img.astype(numpy.float)
    bximg = bximg.astype(numpy.float)

    # check if image are compatible
    if not b0img.shape == bximg.shape:
        raise ArgumentError('The input images shapes differ i.e. {} != {}.'.format(b0img.shape, bximg.shape))
    if not header.get_pixel_spacing(b0hdr) == header.get_pixel_spacing(bxhdr):
        raise ArgumentError('The input images voxel spacing differs i.e. {} != {}.'.format(header.get_pixel_spacing(b0hdr), header.get_pixel_spacing(bxhdr)))
    
    # check if supplied threshold value as well as the b value is above 0
    if args.threshold is not None and not args.threshold >= 0:
        raise ArgumentError('The supplied threshold value must be greater than 0, otherwise a division through 0 might occur.')
    if not args.b > 0:
        raise ArgumentError('The supplied b-value must be greater than 0.')
    
    # compute threshold value if not supplied
    if args.threshold is None:
        b0thr = otsu(b0img, 32) / 4. # divide by 4 to decrease impact
        bxthr = otsu(bximg, 32) / 4.
        if 0 >= b0thr:
            raise ArgumentError('The supplied b0image seems to contain negative values.')
        if 0 >= bxthr:
            raise ArgumentError('The supplied bximage seems to contain negative values.')
    else:
        b0thr = bxthr = args.threshold
    
    logger.debug('thresholds={}/{}, b-value={}'.format(b0thr, bxthr, args.b))
    
    # threshold b0 + bx DW image to obtain a mask
    # b0 mask avoid division through 0, bx mask avoids a zero in the ln(x) computation
    mask = binary_fill_holes(b0img > b0thr) & binary_fill_holes(bximg > bxthr)
    
    # perform a number of binary morphology steps to select the brain only
    mask = binary_erosion(mask, iterations=1)
    mask = largest_connected_component(mask)
    mask = binary_dilation(mask, iterations=1)
    
    logger.debug('excluding {} of {} voxels from the computation and setting them to zero'.format(numpy.count_nonzero(mask), numpy.prod(mask.shape)))
    
    # compute the ADC
    adc = numpy.zeros(b0img.shape, b0img.dtype)
    adc[mask] = -1. * args.b * numpy.log(bximg[mask] / b0img[mask])
    adc[adc < 0] = 0
            
    # saving the resulting image
    save(adc, args.output, b0hdr, args.force)

    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('b0image', help='the diffusion weighted image required with b=0')
    parser.add_argument('bximage', help='the diffusion weighted image required with b=x')
    parser.add_argument('b', type=int, help='the b-value used to acquire the bx-image (i.e. x)')
    parser.add_argument('output', help='the computed apparent diffusion coefficient image')
    
    parser.add_argument('-t', '--threshold', type=int, dest='threshold', help='set a fixed threshold for the input images to mask the computation')
    
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='overwrite existing files')
    return parser
    
if __name__ == "__main__":
    main()        
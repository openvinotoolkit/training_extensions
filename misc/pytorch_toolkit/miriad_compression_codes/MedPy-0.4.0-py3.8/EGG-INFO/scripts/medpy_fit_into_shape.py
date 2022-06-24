#!/usr/bin/python3

"""
Fit an existing image into a new shape.

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
Fit an existing image into a new shape.

If larger, the original image is placed centered in all dimensions. If smaller,
it is cut equally at all sides.

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
    
    # check shape dimensionality
    if not len(args.shape) == img.ndim:
        parser.error('The image has {} dimensions, but {} shape parameters have been supplied.'.format(img.ndim, len(args.shape)))
        
    # check if output image exists
    if not args.force and os.path.exists(args.output):
        parser.error('The output image {} already exists.'.format(args.output))         
    
    # compute required cropping and extention
    slicers_cut = []
    slicers_extend = []
    for dim in range(len(img.shape)):
        slicers_cut.append(slice(None))
        slicers_extend.append(slice(None))
        if args.shape[dim] != img.shape[dim]:
            difference = abs(img.shape[dim] - args.shape[dim])
            cutoff_left = difference / 2
            cutoff_right = difference / 2 + difference % 2
            if args.shape[dim] > img.shape[dim]:
                slicers_extend[-1] = slice(cutoff_left, -1 * cutoff_right)
            else:
                slicers_cut[-1] = slice(cutoff_left, -1 * cutoff_right)
            
    # crop original image
    img = img[slicers_cut]
    
    # create output image and place input image centered
    out = numpy.zeros(args.shape, img.dtype)
    out[slicers_extend] = img
    
    # saving the resulting image
    save(out, args.output, hdr, args.force)
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__description__)
    parser.add_argument('input', help='the input image')
    parser.add_argument('output', help='the output image')
    parser.add_argument('shape', type=argparseu.sequenceOfIntegersGt, help='the desired shape in colon-separated values, e.g. 255,255,32')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='overwrite existing files')
    return parser
    
if __name__ == "__main__":
    main()    

#!/usr/bin/python3

"""
Convert a binary volume into a surface contour.

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
import math

# third-party modules
import numpy
from scipy.ndimage.morphology import binary_erosion, binary_dilation,\
    generate_binary_structure

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2014-06-04"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Converts a binary volume into a surface contour. In the case of an even
                  contour width, the surface of the volume will correspond with the
                  middle of the contour line. In the case of an odd contour width, the
                  contour will be shifted by one voxel towards the inside of the volume.
                  
                  In the case of 3D volumes, the contours result in shells, which might
                  not be desired, as they do not visualize well in 2D views. With the
                  '--dimension' argument, a dimension along which to extract the contours
                  can be supplied.
                  
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
    
    # load input image
    data_input, header_input = load(args.input)
    
    # treat as binary
    data_input = data_input.astype(numpy.bool)
    
    # check dimension argument
    if args.dimension and (not args.dimension >= 0 or not args.dimension < data_input.ndim):
        argparse.ArgumentError(args.dimension, 'Invalid dimension of {} supplied. Image has only {} dimensions.'.format(args.dimension, data_input.ndim))
        
    # compute erosion and dilation steps
    erosions = int(math.ceil(args.width / 2.))
    dilations = int(math.floor(args.width / 2.))
    logger.debug("Performing {} erosions and {} dilations to achieve a contour of width {}.".format(erosions, dilations, args.width))
    
    # erode, dilate and compute contour
    if not args.dimension:
        eroded = binary_erosion(data_input, iterations=erosions) if not 0 == erosions else data_input
        dilated = binary_dilation(data_input, iterations=dilations) if not 0 == dilations else data_input
        data_output = dilated - eroded
    else:
        slicer = [slice(None)] * data_input.ndim
        bs_slicer = [slice(None)] * data_input.ndim
        data_output = numpy.zeros_like(data_input)
        for sl in range(data_input.shape[args.dimension]):
            slicer[args.dimension] = slice(sl, sl+1)
            bs_slicer[args.dimension] = slice(1, 2)
            bs = generate_binary_structure(data_input.ndim, 1)
            
            eroded = binary_erosion(data_input[slicer], structure=bs[bs_slicer], iterations=erosions) if not 0 == erosions else data_input[slicer]
            dilated = binary_dilation(data_input[slicer], structure=bs[bs_slicer], iterations=dilations) if not 0 == dilations else data_input[slicer]
            data_output[slicer] = dilated - eroded
    logger.debug("Contour image contains {} contour voxels.".format(numpy.count_nonzero(data_output)))

    # save resulting volume
    save(data_output, args.output, header_input, args.force)
    
    logger.info("Successfully terminated.")    
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    if args.width <= 0:
        raise argparse.ArgumentError(args.width, 'The contour width must be a positive number.')
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='Source volume.')
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('-w', '--width', dest='width', type=int, default=1, help='Width of the contour.')
    parser.add_argument('--dimension', type=int, help='Extract contours only along this dimension.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()        
#!/usr/bin/python3

"""
Joins a number of XD volumes into a (X+1)D volume.

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
from argparse import RawTextHelpFormatter
import logging

# third-party modules
import scipy

# path changes

# own modules
from medpy.io import load, save, header
from medpy.core import Logger
from medpy.core.exceptions import ArgumentError


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.3, 2012-05-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Joins a number of XD volumes into a (X+1)D volume.
                  
                  One common use is when a number of 3D volumes, each representing a
                  moment in time, are availabel. With this script they can be joined
                  into a proper 4D volume.
                  
                  Copyright (C) 2013 Oskar Maier
                  This program comes with ABSOLUTELY NO WARRANTY; This is free software,
                  and you are welcome to redistribute it under certain conditions; see
                  the LICENSE file or <http://www.gnu.org/licenses/> for details.   
                  """

# code
def main():
    # parse cmd arguments
    parser = getParser()
    parser.parse_args()
    args = getArguments(parser)
    
    # prepare logger
    logger = Logger.getInstance()
    if args.debug: logger.setLevel(logging.DEBUG)
    elif args.verbose: logger.setLevel(logging.INFO)
    
    # load first input image as example 
    example_data, example_header = load(args.inputs[0])
    
    # test if the supplied position is valid
    if args.position > example_data.ndim or args.position < 0:
        raise ArgumentError('The supplied position for the new dimension is invalid. It has to be between 0 and {}.'.format(example_data.ndim))
    
    # prepare empty output volume
    output_data = scipy.zeros([len(args.inputs)] + list(example_data.shape), dtype=example_data.dtype)
    
    # add first image to output volume
    output_data[0] = example_data
    
    # load input images and add to output volume
    for idx, image in enumerate(args.inputs[1:]):
        image_data, _ = load(image)
        if not args.ignore and image_data.dtype != example_data.dtype:
            raise ArgumentError('The dtype {} of image {} differs from the one of the first image {}, which is {}.'.format(image_data.dtype, image, args.inputs[0], example_data.dtype))
        if image_data.shape != example_data.shape:
            raise ArgumentError('The shape {} of image {} differs from the one of the first image {}, which is {}.'.format(image_data.shape, image, args.inputs[0], example_data.shape))
        output_data[idx + 1] = image_data
        
    # move new dimension to the end or to target position
    for dim in range(output_data.ndim - 1):
        if dim >= args.position: break
        output_data = scipy.swapaxes(output_data, dim, dim + 1)
        
    # set pixel spacing
    spacing = list(header.get_pixel_spacing(example_header))
    spacing = tuple(spacing[:args.position] + [args.spacing] + spacing[args.position:])
    example_header.set_voxel_spacing(spacing)
    
    # save created volume
    save(output_data, args.output, example_header, args.force)
        
    logger.info("Successfully terminated.")
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('inputs', nargs='+', help='Source volumes of same shape and dtype.')
    parser.add_argument('-s', dest='spacing', type=float, default=1, help='The voxel spacing of the newly created dimension. Default is 1.')
    parser.add_argument('-p', dest='position', type=int, default=0, help='The position where to put the new dimension starting from 0. Standard behaviour is to place it in the first position.')
    parser.add_argument('-i', dest='ignore', action='store_true', help='Ignore if the images datatypes differ.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()

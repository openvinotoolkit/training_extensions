#!/usr/bin/python3

"""
Splits a XD into a number of (X-1)D volumes.

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
import scipy

# path changes

# own modules
from medpy.io import load, save, header
from medpy.core import Logger
from medpy.core.exceptions import ArgumentError


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.2, 2012-05-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Splits a XD into a number of (X-1)D volumes.
                  
                  One common use case is the creation of manual markers for 4D images.
                  This script allows to split a 4D into a number of either spatial or
                  temporal 3D volumes, for which one then can create the markers. These
                  can be rejoined using the join_xd_to_xplus1d.py script.
                  
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
    
    # load input image
    data_input, header_input = load(args.input)
    
    # check if the supplied dimension is valid
    if args.dimension >= data_input.ndim or args.dimension < 0:
        raise ArgumentError('The supplied cut-dimension {} exceeds the image dimensionality of 0 to {}.'.format(args.dimension, data_input.ndim - 1))
    
    # prepare output file string
    name_output = args.output.replace('{}', '{:03d}')
    
    # compute the new the voxel spacing
    spacing = list(header.get_pixel_spacing(header_input))
    del spacing[args.dimension]
    
    # iterate over the cut dimension
    slices = data_input.ndim * [slice(None)]
    for idx in range(data_input.shape[args.dimension]):
        # cut the current slice from the original image 
        slices[args.dimension] = slice(idx, idx + 1)
        data_output = scipy.squeeze(data_input[slices])
        # update the header and set the voxel spacing
        header_input.set_voxel_spacing(spacing)
        # save current slice
        save(data_output, name_output.format(idx), header_input, args.force)
        
    logger.info("Successfully terminated.")
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    if not '{}' in args.output:
        raise argparse.ArgumentError(args.output, 'The output argument string must contain the sequence "{}".')
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='Source volume.')
    parser.add_argument('output', help='Target volumes. Has to include the sequence "{}" in the place where the volume number should be placed.')
    parser.add_argument('dimension', type=int, help='The dimension along which to split (starting from 0).')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()

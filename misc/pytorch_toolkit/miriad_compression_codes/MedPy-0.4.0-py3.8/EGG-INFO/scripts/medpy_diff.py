#!/usr/bin/python3

"""
Compares the pixel values of two images and gives a measure of the difference.

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
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

# build-in modules
import sys
import argparse
import logging

# third-party modules
import scipy

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load
from functools import reduce


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2012-05-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Compares the pixel values of two images and gives a measure of the difference.
                  
                  Also compares the dtype and shape.
                  
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
    
    # load input image1
    data_input1, _ = load(args.input1)
    
    # load input image2
    data_input2, _ = load(args.input2)
    
    # compare dtype and shape
    if not data_input1.dtype == data_input2.dtype: print('Dtype differs: {} to {}'.format(data_input1.dtype, data_input2.dtype))
    if not data_input1.shape == data_input2.shape:
        print('Shape differs: {} to {}'.format(data_input1.shape, data_input2.shape))
        print('The voxel content of images of different shape can not be compared. Exiting.')
        sys.exit(-1)
    
    # compare image data
    voxel_total = reduce(lambda x, y: x*y, data_input1.shape)
    voxel_difference = len((data_input1 != data_input2).nonzero()[0])
    if not 0 == voxel_difference:
        print('Voxel differ: {} of {} total voxels'.format(voxel_difference, voxel_total))
        print('Max difference: {}'.format(scipy.absolute(data_input1 - data_input2).max()))
    else: print('No other difference.')
    
    logger.info("Successfully terminated.")    
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input1', help='Source volume one.')
    parser.add_argument('input2', help='Source volume two.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()        
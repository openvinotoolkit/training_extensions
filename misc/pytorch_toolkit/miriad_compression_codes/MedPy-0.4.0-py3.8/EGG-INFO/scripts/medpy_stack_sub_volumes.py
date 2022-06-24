#!/usr/bin/python3

"""
Stacks a number of volumes into one dimension.

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
from argparse import RawTextHelpFormatter
import argparse
import logging

# third-party modules
import numpy

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save

# information
__author__ = "Oskar Maier"
__version__ = "r0.3.1, 2011-03-29"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Takes a arbitrary number of medical images that are of equal depth in
                  all but one dimension. The images are then stacked on top of each other
                  to produce a single result image. The dimension in which to stack is
                  supplied by the dimension parameter.
                  
                  Note that the supplied images must be of the same data type.
                  Note to take into account the input images orientations.
                  
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
    
    # load first image as result image
    logger.info('Loading {}...'.format(args.images[0]))
    result_data, result_header = load(args.images[0])
    
    # check dimension argument
    if args.dimension >= result_data.ndim:
        raise argparse.ArgumentError('The supplied stack-dimension {} exceeds the image dimensionality of 0 to {}.'.format(args.dimension, result_data.ndim - 1))
    
    # reduce the image dimensions
    if args.zero and result_data.all():
        result_data = numpy.zeros(result_data.shape, result_data.dtype)
    
    # iterate over remaining images and concatenate
    for image_name in args.images[1:]:
        logger.info('Loading {}...'.format(image_name))
        image_data, _ = load(image_name)
        
        # change to zero matrix if requested
        if args.zero and image_data.all():
            image_data = numpy.zeros(image_data.shape, image_data.dtype)
        
        #concatenate
        if args.reversed:
            result_data = numpy.concatenate((image_data, result_data), args.dimension)
        else: 
            result_data = numpy.concatenate((result_data, image_data), args.dimension)

    logger.debug('Final image is of shape {}.'.format(result_data.shape))

    # save results in same format as input image
    logger.info('Saving concatenated image as {}...'.format(args.output))
    
    save(result_data, args.output, result_header, args.force)
    
    logger.info('Successfully terminated.')
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('dimension', type=int, help='The dimension in which direction to stack (starting from 0:x).')
    parser.add_argument('output', help='The output image.')
    parser.add_argument('images', nargs='+', help='The images to concatenate/stack.')
    parser.add_argument('-f', dest='force', action='store_true', help='Set this flag to silently override files that exist.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-z', dest='zero', action='store_true', help='If supplied, all images containing only 1s are treated as empty image.')
    parser.add_argument('-r', dest='reversed', action='store_true', help='Stack in resversed order as how the files are supplied.')
    
    return parser    
    
if __name__ == "__main__":
    main()    
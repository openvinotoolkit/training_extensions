#!/usr/bin/python3

"""
Joins a number of binary images into a single conjunction.

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
import numpy

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save, header


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2014-05-15"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Joins a number of binary images into a single conjunction.
                  
                  The available combinatorial operations are sum, avg, max and min.
                  In the case of max and min, the output volumes are also binary images,
                  in the case of sum they are uint8 and in the case of avg of type float.
                  
                  All input images must be of same shape and voxel spacing.
                  
                  WARNING: Does not consider image offset.
                  
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
    
    
    # load input images and cast to bool
    images = []
    for input_ in args.inputs:
        t = load(input_)
        images.append((t[0], t[1]))
    
    # check if their shapes and voxel spacings are all equal
    s0 = images[0][0].shape
    if not numpy.all([i[0].shape == s0 for i in images[1:]]):
        raise argparse.ArgumentError(args.input, 'At least one input image is of a different shape than the others.')
    vs0 = header.get_pixel_spacing(images[0][1])
    if not numpy.all([header.get_pixel_spacing(i[1]) == vs0 for i in images[1:]]):
        raise argparse.ArgumentError(args.input, 'At least one input image has a different voxel spacing than the others.')
    
    # execute operation
    logger.debug('Executing operation {} over {} images.'.format(args.operation, len(images)))
    if 'max' == args.operation:
        out = numpy.maximum.reduce([t[0] for t in images])
    elif 'min' == args.operation:
        out = numpy.minimum.reduce([t[0] for t in images])
    elif 'sum' == args.operation:
        out = numpy.sum([t[0] for t in images], 0).astype(numpy.uint8)
    else: # avg
        out = numpy.average([t[0] for t in images], 0).astype(numpy.float32)
        
    # save output
    save(out, args.output, images[0][1], args.force)
    
    logger.info("Successfully terminated.")    
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__description__)
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('inputs', nargs='+', help='Source volume(s).')
    parser.add_argument('-o', '--operation', dest='operation', choices=['sum', 'avg', 'max', 'min'], default='avg', help='Combinatorial operation to conduct.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()        
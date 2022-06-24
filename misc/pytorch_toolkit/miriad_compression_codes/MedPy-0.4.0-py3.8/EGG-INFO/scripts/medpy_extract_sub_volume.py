#!/usr/bin/python3

"""
Extracts a sub-volume from a medical image.

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
import sys
import os

# third-party modules
import scipy

# path changes

# own modules
from medpy.core import ArgumentError, Logger
from medpy.io import load, save

# information
__author__ = "Oskar Maier"
__version__ = "r0.3.0, 2011-12-11"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Takes a medical image of arbitrary dimensions and the dimensions
                  of a sub-volume that lies inside the dimensions of this images.
                  Extracts the sub-volume from the supplied image and saves it.
                  
                  The volume to be extracted is defined by its slices, the syntax is the same as
                  for numpy array indexes (i.e. starting with zero-index, the first literal (x) of any
                  x:y included and the second (y) excluded).
                  E.g. '2:3,4:6' would extract the slice no. 3 in X and 5, 6 in Y direction of a 2D image.
                  E.g. '99:199,149:199,99:249' would extract the respective slices in X,Y and Z direction of a 3D image.
                       This could, for example, be used to extract the area of the liver form a CT scan.
                  To keep all slices in one direction just omit the respective value:
                  E.g. '99:199,149:199,' would work ust as example II, but extract all Z slices.
                       Note here the trailing colon.

                  Note to take into account the input images orientation when supplying the sub-volume.
                  
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
    
    # check if output image exists
    if not args.force:
        if os.path.exists(args.output + args.image[-4:]):
            logger.warning('The output file {} already exists. Breaking.'.format(args.output + args.image[-4:]))
            exit(1)
    
    # load images
    image_data, image_header = load(args.image)
    
    # check image dimensions against sub-volume dimensions
    if len(image_data.shape) != len(args.volume):
        logger.critical('The supplied input image is of different dimension as the sub volume requested ({} to {})'.format(len(image_data.shape), len(args.volume)))
        raise ArgumentError('The supplied input image is of different dimension as the sub volume requested ({} to {})'.format(len(image_data.shape), len(args.volume)))
    
    # execute extraction of the sub-area  
    logger.info('Extracting sub-volume...')
    index = [slice(x[0], x[1]) for x in args.volume]
    volume = image_data[index]
    
    # check if the output image contains data
    if 0 == len(volume):
        logger.exception('The extracted sub-volume is of zero-size. This usual means that the supplied volume coordinates and the image coordinates do not intersect. Exiting the application.')
        sys.exit(-1)
    
    # squeeze extracted sub-volume for the case in which one dimensions has been eliminated
    volume = scipy.squeeze(volume)
    
    logger.debug('Extracted volume is of shape {}.'.format(volume.shape))
    
    # save results in same format as input image
    save(volume, args.output, image_header, args.force)
    
    logger.info('Successfully terminated.')

    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    # parse volume and adapt to zero-indexing
    try:
        def _to_int_or_none(string):
            if 0 == len(string): return None
            return int(string)
        def _to_int_or_none_double (string):
            if 0 == len(string): return [None, None]
            return list(map(_to_int_or_none, string.split(':')))        
        args.volume = list(map(_to_int_or_none_double, args.volume.split(',')))
        args.volume = [(x[0], x[1]) for x in args.volume]
    except (ValueError, IndexError) as e:
        raise ArgumentError('Maleformed volume parameter "{}", see description with -h flag.'.format(args.volume), e)

    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('image', help='The source volume.')
    parser.add_argument('output', help='The target volume.')
    parser.add_argument('volume', help='The coordinated of the sub-volume of the images that should be extracted.\nExample: 30:59,40:67,45:75 for a 3D image.\nSee -h for more information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Set this flag to silently override files that exist.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    
    return parser    
    
if __name__ == "__main__":
    main()    
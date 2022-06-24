#!/usr/bin/python3

"""
Extracts a sub-volume from a medical image by an example image.

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
import numpy

# path changes

# own modules
from medpy.core import ArgumentError, Logger
from medpy.io import load, save

# information
__author__ = "Oskar Maier"
__version__ = "r0.2.0, 2011-12-11"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Takes a medical image of arbitrary dimensions and a binary mask
                  image of the same dimensions. Extract the exact position of the
                  binary mask in the binary mask image and uses these dimensions
                  for the extraction of a sub-volume that lies inside the dimensions
                  of the medical images.
                  Extracts the sub-volume from the supplied image and saves it.
                  
                  Note that both images must be of the same dimensionality, otherwise an exception is thrown.
                  Note that the input images offset is not taken into account.
                  Note to take into account the input images orientation.
                  
                  This is a convenience script, combining the functionalities of
                  extract_mask_position and extract_sub_volume.
                  
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
    
    # load mask
    logger.info('Loading mask {}...'.format(args.mask))
    mask_image, _ = load(args.mask)
    
    # store mask images shape for later check against the input image
    mask_image_shape = mask_image.shape 
    
    # extract the position of the foreground object in the mask image
    logger.info('Extract the position of the foreground object...')
    positions = mask_image.nonzero()
    positions = [(max(0, positions[i].min() - args.offset), positions[i].max() + 1 + args.offset)
                    for i in range(len(positions))] # crop negative values
    logger.debug('Extracted position is {}.'.format(positions))

    # load image
    logger.info('Loading image {}...'.format(args.image))
    image_data, image_header = load(args.image)
    
    # check if the mask image and the input image are of the same shape
    if mask_image_shape != image_data.shape:
        raise ArgumentError('The two input images are of different shape (mask: {} and image: {}).'.format(mask_image_shape, image_data.shape))
    
    # execute extraction of the sub-area  
    logger.info('Extracting sub-volume...')
    index = tuple([slice(x[0], x[1]) for x in positions])
    volume = image_data[index]
    
    # check if the output image contains data
    if 0 == len(volume):
        logger.exception('The extracted sub-volume is of zero-size. This usual means that the mask image contained no foreground object.')
        sys.exit(0)
    
    logger.debug('Extracted volume is of shape {}.'.format(volume.shape))
    
    # get base origin of the image
    origin_base = numpy.array([0] * image_data.ndim) # for backwards compatibility
        
    # modify the volume offset to imitate numpy behavior (e.g. wrap negative values)
    offset = numpy.array([x[0] for x in positions])
    for i in range(0, len(offset)):
        if None == offset[i]: offset[i] = 0
    offset[offset<0] += numpy.array(image_data.shape)[offset<0] # wrap around
    offset[offset<0] = 0 # set negative to zero
    
    # calculate final new origin
    origin = origin_base + offset
    
    logger.debug('Final origin created as {} + {} = {}.'.format(origin_base, offset, origin))
    
    # save results in same format as input image
    logger.info('Saving extracted volume...')
    save(volume, args.output, image_header, args.force)
    
    logger.info('Successfully terminated.')

    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()    
    # check output image exists if override not forced
    if not args.force:
        if os.path.exists(args.output + args.image[-4:]):
            raise ArgumentError('The supplied output file {} already exists. Run -f/force flag to override.'.format(args.output))

    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('image', help='The input image.')
    parser.add_argument('output', help='The resulting sub-volume.')
    parser.add_argument('mask', help='A mask image containing a single foreground object (non-zero).')
    parser.add_argument('-o', '--offset', dest='offset', default=0, type=int, help='Set an offset by which the extracted sub-volume size should be increased in all directions.')
    parser.add_argument('-f', dest='force', action='store_true', help='Set this flag to silently override files that exist.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    
    return parser    
    
if __name__ == "__main__":
    main()    

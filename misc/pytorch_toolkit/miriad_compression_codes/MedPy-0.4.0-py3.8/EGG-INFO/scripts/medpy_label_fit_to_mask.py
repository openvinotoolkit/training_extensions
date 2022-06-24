#!/usr/bin/python3

"""
Executes a reduce operation taking a mask and a number of label images as input.

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
import os

# third-party modules
import numpy

# path changes

# own modules
from medpy.io import load, save 
from medpy.core import Logger
from medpy.filter import fit_labels_to_mask


# information
__author__ = "Oskar Maier"
__version__ = "r0.2.0, 2011-12-12"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Reduces label images by fitting them as best as possible to a supplied
                  mask and subsequently creating mask out of them.
                  The resulting image is saved in the supplied folder with the same
                  name as the input image, but with a suffix '_reduced' attached.
                  For each region the intersection with the reference mask is computed
                  and if the value exceeds 50% of the total region size, it is marked
                  as mask, otherwise as background. For more details on how the fitting
                  is performed @see filter.fit_labels_to_mask.
                  
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
    logger.info('Loading image {}...'.format(args.input))
    image_labels_data, _ = load(args.image)    
    
    # load mask image
    logger.info('Loading mask {}...'.format(args.mask))
    image_mask_data, image_mask_data_header = load(args.mask)
    
    # check if output image exists
    if not args.force:
        if os.path.exists(args.output):
            logger.warning('The output image {} already exists. Skipping this image.'.format(args.output))
    
    # create a mask from the label image
    logger.info('Reducing the label image...')
    image_reduced_data = fit_labels_to_mask(image_labels_data, image_mask_data)
    
    # save resulting mask
    logger.info('Saving resulting mask as {} in the same format as input mask, only with data-type int8...'.format(args.output))
    image_reduced_data = image_reduced_data.astype(numpy.bool, copy=False) # bool sadly not recognized
    save(image_reduced_data, args.output, image_mask_data_header, args.force)
    
    logger.info('Successfully terminated.')
        
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)

    parser.add_argument('image', nargs='+', help='The input label image.')
    parser.add_argument('mask', help='The mask image to which to fit the label images.')
    parser.add_argument('output', help='The output image.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    
    return parser    
    
if __name__ == "__main__":
    main()        

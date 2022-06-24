#!/usr/bin/python3

"""
Creates the superimposition image of two label images.

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
from argparse import ArgumentError
import argparse
import logging
import os

# third-party modules
import scipy

# path changes

# own modules
from medpy.io import load, save
from medpy.core import Logger

# information
__author__ = "Oskar Maier"
__version__ = "r0.2.1, 2011-01-04"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Takes two label images as input and creates their superimposition i.e.
                  all the regions borders are preserved and the resulting image contains
                  more or the same number of regions as the respective input images.
                  
                  The resulting image has the same name as the first input image, just
                  with a '_superimp' suffix.
                  
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

    # build output image name
    image_superimposition_name = args.folder + '/' + args.image1.split('/')[-1][:-4] + '_superimp'
    image_superimposition_name += args.image1.split('/')[-1][-4:]
        
    # check if output image exists
    if not args.force:
        if os.path.exists(image_superimposition_name):
            raise ArgumentError('The output image {} already exists. Please provide the -f/force flag, if you wish to override it.'.format(image_superimposition_name))
    
    # load image1 using
    logger.info('Loading image {}...'.format(args.image1))
    image1_data, image1_header = load(args.image1)
    
    # load image2 using
    logger.info('Loading image {}...'.format(args.image2))
    image2_data, _ = load(args.image2)
        
    # check input images to be valid
    logger.info('Checking input images for correctness...')
    if image1_data.shape != image2_data.shape:
        raise ArgumentError('The two input images shape do not match with 1:{} and 2:{}'.format(image1_data.shape, image2_data.shape))
    int_types = (scipy.uint, scipy.uint0, scipy.uint8, scipy.uint16, scipy.uint32, scipy.uint64, scipy.uintc, scipy.uintp,
                 scipy.int_, scipy.int0, scipy.int8, scipy.int16, scipy.int32, scipy.int64, scipy.intc, scipy.intp)
    if image1_data.dtype not in int_types:
        raise ArgumentError('Input image 1 is of type {}, an int type is required.'.format(image1_data.dtype))
    if image2_data.dtype not in int_types:
        raise ArgumentError('Input image 2 is of type {}, an int type is required.'.format(image2_data.dtype))
    if 4294967295 < abs(image1_data.min()) + image1_data.max() + abs(image2_data.min()) + image2_data.max():
        raise ArgumentError('The input images contain so many (or not consecutive) labels, that they will not fit in a uint32 range.')
        
    # create superimposition of the two label images
    logger.info('Creating superimposition image...')
    image_superimposition_data = scipy.zeros(image1_data.shape, dtype=scipy.uint32)
    translation = {}
    label_id_counter = 0
    for x in range(image1_data.shape[0]):
        for y in range(image1_data.shape[1]):
            for z in range(image1_data.shape[2]):
                label1 = image1_data[x,y,z]
                label2 = image2_data[x,y,z]
                if not (label1, label2) in translation:
                    translation[(label1, label2)] = label_id_counter
                    label_id_counter += 1
                image_superimposition_data[x,y,z] = translation[(label1, label2)]
    
    # save resulting superimposition image
    logger.info('Saving superimposition image as {} in the same format as input image...'.format(image_superimposition_name))
    save(image_superimposition_data, args.output, image1_header, args.force)
    
    logger.info('Successfully terminated.')
        
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('image1', help='The first input label image.')
    parser.add_argument('image2', help='The second input label image.')
    parser.add_argument('output', help='The output image.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    
    return parser    
    
if __name__ == "__main__":
    main()        

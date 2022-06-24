#!/usr/bin/python3

"""
Automatically extracts sub-volumes from a medical image.

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
import os

# third-party modules

# path changes

# own modules
from medpy.core import ArgumentError, Logger
from medpy.io import load, save

# information
__author__ = "Oskar Maier"
__version__ = "r0.2.1, 2012-05-17"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Takes a medical image of arbitrary dimensions and splits it into a
                  number of sub-volumes along the supplied dimensions. The maximum size
                  of each such created volume can be supplied.
                  
                  Note to take into account the input images orientation when supplying the cut dimension.
                  Note that the image offsets are not preserved.
                  
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
    logger.info('Loading {}...'.format(args.image))
    image_data, image_header = load(args.image)
    
    # check if supplied cut dimension is inside the input images dimensions
    if args.dimension < 0 or args.dimension >= image_data.ndim:
        logger.critical('The supplied cut-dimensions {} is invalid. The input image has only {} dimensions.'.format(args.dimension, image_data.ndim))
        raise ArgumentError('The supplied cut-dimensions {} is invalid. The input image has only {} dimensions.'.format(args.dimension, image_data.ndim))
    
    # prepare output filenames
    name_output = args.output.replace('{}', '{:03d}')
    
    # determine cut lines
    no_sub_volumes = image_data.shape[args.dimension] / args.maxsize + 1 # int-division is desired
    slices_per_volume = image_data.shape[args.dimension] / no_sub_volumes # int-division is desired
    
    # construct processing dict for each sub-volume
    processing_array = []
    for i in range(no_sub_volumes):
        processing_array.append(
            {'path': name_output.format(i+1),
             'cut': (i * slices_per_volume, (i + 1) * slices_per_volume)})
        if no_sub_volumes - 1 == i: # last volume has to have increased cut end
            processing_array[i]['cut'] = (processing_array[i]['cut'][0], image_data.shape[args.dimension])

    # construct base indexing list
    index = [slice(None) for _ in range(image_data.ndim)]
    
    # execute extraction of the sub-volumes
    logger.info('Extracting sub-volumes...')
    for dic in processing_array:
        # check if output images exists
        if not args.force:
            if os.path.exists(dic['path']):
                logger.warning('The output file {} already exists. Skipping this volume.'.format(dic['path']))
                continue
        
        # extracting sub-volume
        index[args.dimension] = slice(dic['cut'][0], dic['cut'][1])
        volume = image_data[index]
        
        logger.debug('Extracted volume is of shape {}.'.format(volume.shape))
        
        # saving sub-volume in same format as input image
        logger.info('Saving cut {} as {}...'.format(dic['cut'], dic['path']))
        save(volume, dic['path'], image_header, args.force)
        
    logger.info('Successfully terminated.')

    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('image', help='An image of arbitrary dimensions that should be split.')
    parser.add_argument('output', help='Output volumes. Has to include the sequence "{}" in the place where the volume number should be placed.')
    parser.add_argument('dimension', type=int, help='The dimension in which direction to split (starting from 0:x).')
    parser.add_argument('maxsize', type=int, help='The produced volumes will always be smaller than this size (in terms of slices in the cut-dimension).')
    parser.add_argument('-f', dest='force', action='store_true', help='Set this flag to silently override files that exist.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    
    return parser    
    
if __name__ == "__main__":
    main()    

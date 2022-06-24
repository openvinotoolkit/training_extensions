#!/usr/bin/python3

"""
Extracts the intersecting parts of two volumes regarding offset and voxel-spacing. 

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

# path changes

# own modules
from medpy.io import load, save, header
from medpy.core import Logger
from medpy.filter.utilities import intersection


# information
__author__ = "Oskar Maier"
__version__ = "r0.0.1, 2014-04-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Development"
__description__ = """
                  Extracts the intersecting parts of two volumes regarding offset
                  and voxel-spacing.
                                  
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
    
    # check if output image exists (will also be performed before saving, but as the smoothing might be very time intensity, a initial check can save frustration)
    if not args.force:
        if os.path.exists(args.output1):
            raise parser.error('The output image {} already exists.'.format(args.output1))
        if os.path.exists(args.output2):
            raise parser.error('The output image {} already exists.'.format(args.output2))
    
    # loading images
    data_input1, header_input1 = load(args.input1)
    data_input2, header_input2 = load(args.input2)
    logger.debug('Original image sizes are {} and {}.'.format(data_input1.shape, data_input2.shape))
    
    # compute intersection volumes (punch)
    logger.info('Computing the intersection.')
    inters1, inters2, new_offset = intersection(data_input1, header_input1, data_input2, header_input2)
    logger.debug('Punched images are of sizes {} and {} with new offset {}.'.format(inters1.shape, inters2.shape, new_offset))
    
    # check if any intersection could be found at all
    if 0 == inters1.size:
        logger.warning('No intersection could be found between the images. Please check their meta-data e.g. with medpy_info')
    
    # update header informations
    header.set_offset(header_input1, new_offset)
    header.set_offset(header_input2, new_offset)
    
    # save punched images
    save(inters1, args.output1, header_input1, args.force)
    save(inters2, args.output2, header_input2, args.force)
    
    logger.info('Successfully terminated.')

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input1', help='First source volume.')
    parser.add_argument('input2', help='Second source volume.')
    parser.add_argument('output1', help='First target volume.')
    parser.add_argument('output2', help='Second target volume.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    
    return parser
    
if __name__ == "__main__":
    main()

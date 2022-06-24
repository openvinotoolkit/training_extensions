#!/usr/bin/python3

"""
Takes a number of label images and counts their regions.

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
import argparse
import logging
import sys

# third-party modules
import numpy

# path changes

# own modules
from medpy.io import load
from medpy.core import Logger


# information
__author__ = "Oskar Maier"
__version__ = "r0.2, 2011-12-13"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Counts the regions in a number of label images and prints the results
                  to the stdout in csv syntax.
                  
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
        
    # write header line
    print('image;labels\n')
    
    # iterate over input images
    for image in args.images:
        
        # get and prepare image data
        logger.info('Processing image {}...'.format(image))
        image_data, _ = load(image)
        
        # count number of labels and flag a warning if they reach the ushort border
        count = len(numpy.unique(image_data)) 
        
        # count number of labels and write
        print('{};{}\n'.format(image.split('/')[-1], count))
        
        sys.stdout.flush()
            
    logger.info('Successfully terminated.')
      
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('images', nargs='+', help='One or more label images.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    
    return parser    
    
if __name__ == "__main__":
    main()            
    
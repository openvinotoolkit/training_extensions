#!/usr/bin/python3

"""
Loads an image and saves it with two dimensions swapped.

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

# third-party modules
import scipy

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save, header
from medpy.core.exceptions import ArgumentError


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2012-05-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Two of the input images dimensions are swapped. A (200,100,10) image
                  can such be turned into a (200,10,100) one.
                  
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
    
    # load input image
    data_input, header_input = load(args.input)
    
    logger.debug('Original shape = {}.'.format(data_input.shape))
    
    # check if supplied dimension parameters is inside the images dimensions
    if args.dimension1 >= data_input.ndim or args.dimension1 < 0:
        raise ArgumentError('The first swap-dimension {} exceeds the number of input volume dimensions {}.'.format(args.dimension1, data_input.ndim))
    elif args.dimension2 >= data_input.ndim or args.dimension2 < 0:
        raise ArgumentError('The second swap-dimension {} exceeds the number of input volume dimensions {}.'.format(args.dimension2, data_input.ndim))
    
    # swap axes
    data_output = scipy.swapaxes(data_input, args.dimension1, args.dimension2)
    # swap pixel spacing and offset
    ps = list(header.get_pixel_spacing(header_input))
    ps[args.dimension1], ps[args.dimension2] = ps[args.dimension2], ps[args.dimension1]
    header.set_pixel_spacing(header_input, ps)
    os = list(header.get_offset(header_input))
    os[args.dimension1], os[args.dimension2] = os[args.dimension2], os[args.dimension1]
    header.set_offset(header_input, os)
    
    logger.debug('Resulting shape = {}.'.format(data_output.shape))
    
    # save resulting volume
    save(data_output, args.output, header_input, args.force)
    
    logger.info("Successfully terminated.")    
    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='Source volume.')
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('dimension1', type=int, help='First dimension to swap (starting from 0).')
    parser.add_argument('dimension2', type=int, help='Second dimension to swap (starting from 0).')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()        
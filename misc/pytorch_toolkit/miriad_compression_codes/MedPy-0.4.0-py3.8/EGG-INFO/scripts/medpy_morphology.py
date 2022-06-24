#!/usr/bin/python3

"""
Executes opening and closing morphological operations over the input image(s).

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
import scipy.ndimage.morphology

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save


# information
__author__ = "Oskar Maier"
__version__ = "r2.0.0, 2011-12-13"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Executes opening and closing morphological operations over the input image(s).
                  
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
    image_smoothed_data, image_header = load(args.input)
        
    # perform opening resp. closing
    # in 3D case: size 1 = 6-connectedness, 2 = 12-connectedness, 3 = 18-connectedness, etc.
    footprint = scipy.ndimage.morphology.generate_binary_structure(image_smoothed_data.ndim, args.size)
    if 'erosion' == args.type:
        logger.info('Applying erosion...')
        image_smoothed_data = scipy.ndimage.morphology.binary_erosion(image_smoothed_data, footprint, iterations=args.iterations)
    elif 'dilation' == args.type:
        logger.info('Applying dilation...')
        image_smoothed_data = scipy.ndimage.morphology.binary_dilation(image_smoothed_data, footprint, iterations=args.iterations)
    elif 'opening' == args.type:
        logger.info('Applying opening...')
        image_smoothed_data = scipy.ndimage.morphology.binary_opening(image_smoothed_data, footprint, iterations=args.iterations)
    else: # closing
        logger.info('Applying closing...')
        image_smoothed_data = scipy.ndimage.morphology.binary_closing(image_smoothed_data, footprint, iterations=args.iterations)

    # apply additional hole closing step
    logger.info('Closing holes...')
    image_smoothed_data = scipy.ndimage.morphology.binary_fill_holes(image_smoothed_data)

    # save resulting mas
    save(image_smoothed_data, args.output, image_header, args.force)
            
    logger.info('Successfully terminated.')
      
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='Source volume.')
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('-t', '--type', dest='type', choices=['erosion', 'dilation', 'opening', 'closing'], default='erosion', help='The type of the morphological operation.')
    parser.add_argument('-i', '--iterations', dest='iterations', default=0, type=int, help='The number of iteration to execute. Supply a value of 1 or higher to restrict the effect of the morphological operation. Otherwise it is applied until saturation.')
    parser.add_argument('-s', '--size', dest='size', default=3, type=int, help='Size of the closing element (>=1). The higher this value, the bigger the wholes that get closed (closing) resp. unconnected elements that are removed (opening). In the 3D case, 1 equals a 6-connectedness, 2 a 12-connectedness, 3 a 18-connectedness, etc.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    
    return parser    
    
if __name__ == "__main__":
    main()            
    
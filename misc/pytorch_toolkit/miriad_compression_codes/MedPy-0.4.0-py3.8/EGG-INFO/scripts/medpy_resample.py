#!/usr/bin/python3

"""
Resamples an image according to a supplied voxel spacing.

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
import os
import argparse
import logging

# third-party modules
import scipy.ndimage.interpolation

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save, header
from medpy.utilities import argparseu


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2013-07-08"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
          Resamples an image according to a supplied voxel spacing.

          BSpline is used for interpolation. A order between 1 and 5 can be selected.
          
          Note that the pixel data type of the input image is respected, i.e. a integer
          input image leads to an integer output image etc.

          Copyright (C) 2013 Oskar Maier
          This program comes with ABSOLUTELY NO WARRANTY; This is free software,
          and you are welcome to redistribute it under certain conditions; see
          the LICENSE file or <http://www.gnu.org/licenses/> for details.   
                  """

# code
def main():
    parser = getParser()
    args = getArguments(parser)

    # prepare logger
    logger = Logger.getInstance()
    if args.debug: logger.setLevel(logging.DEBUG)
    elif args.verbose: logger.setLevel(logging.INFO)
    
    # loading input images
    img, hdr = load(args.input)

    # check spacing values
    if not len(args.spacing) == img.ndim:
        parser.error('The image has {} dimensions, but {} spacing parameters have been supplied.'.format(img.ndim, len(args.spacing)))
        
    # check if output image exists
    if not args.force:
        if os.path.exists(args.output):
            parser.error('The output image {} already exists.'.format(args.output)) 
        
    logger.debug('target voxel spacing: {}'.format(args.spacing))

    # compute zoom values
    zoom_factors = [old / float(new) for new, old in zip(args.spacing, header.get_pixel_spacing(hdr))]
    logger.debug('zoom-factors: {}'.format(zoom_factors))

    # zoom image
    img = scipy.ndimage.interpolation.zoom(img, zoom_factors, order=args.order)
    logger.debug('new image shape: {}'.format(img.shape))

    # set new voxel spacing
    header.set_pixel_spacing(hdr, args.spacing)

    # saving the resulting image
    save(img, args.output, hdr, args.force)

    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    if args.order < 0 or args.order > 5:
        parser.error('The order has to be a number between 0 and 5.')   
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='the input image')
    parser.add_argument('output', help='the output image')
    parser.add_argument('spacing', type=argparseu.sequenceOfFloatsGt, help='the desired voxel spacing in colon-separated values, e.g. 1.2,1.2,5.0')
    parser.add_argument('-o', '--order', type=int, default=2, dest='order', help='the bspline order, default is 2;  means nearest neighbours; see also medpy_binary_resampling.py')
    
    #group = parser.add_mutually_exclusive_group(required=False)
    #group.add_argument('--binary', action='store_true', dest='binary', help='enforce binary output image')
    #group.add_argument('--float', action='store_true', dest='float', help='enforce floating point output image')
    
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='overwrite existing files')
    return parser
    
if __name__ == "__main__":
    main()        

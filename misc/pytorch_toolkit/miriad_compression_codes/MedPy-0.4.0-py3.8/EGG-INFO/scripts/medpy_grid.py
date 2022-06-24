#!/usr/bin/python3

"""
Creates an image volume containing a regular grid that can e.g. be used to visualize
deformation fields.

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
import tempfile

# third-party modules
import scipy

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save, header

# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2013-06-13"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Create an image volume containing a regular grid that can e.g. be used to
                  visualize deformation fields. The grid volume can be generated either by
                  supplying an example volume (-e) or by directly defining its shape (-s).
                  
                  Copyright (C) 2013 Oskar Maier
                  This program comes with ABSOLUTELY NO WARRANTY; This is free software,
                  and you are welcome to redistribute it under certain conditions; see
                  the LICENSE file or <http://www.gnu.org/licenses/> for details.   
                  """

def main():
    args = getArguments(getParser())

    # prepare logger
    logger = Logger.getInstance()
    if args.debug: logger.setLevel(logging.DEBUG)
    elif args.verbose: logger.setLevel(logging.INFO)
    
    # copy the example image or generate empty image, depending on the modus
    if args.example:
        grid_image = scipy.zeros(args.example_image.shape, scipy.bool_)
        grid_header = args.example_header
    else:
        grid_image = scipy.zeros(args.shape, scipy.bool_)
        # !TODO: Find another solution for this
        # Saving and loading image once to generate a valid header
        tmp_dir = tempfile.mkdtemp()
        tmp_image = '{}/{}'.format(tmp_dir, args.output.split('/')[-1])
        save(grid_image, tmp_image)
        _, grid_header = load(tmp_image)
        try:
            os.remove(tmp_image)
            os.rmdir(tmp_dir)
        except Exception:
            pass
        
    # set the image attributes if supplied
    if args.pixelspacing:
        header.set_pixel_spacing(grid_header, args.pixelspacing)
    if args.offset:
        header.set_offset(grid_header, args.offset)
    
    # compute the right grid spacing for each dimension
    if args.real:
        grid_spacing = [int(round(sp / float(ps))) for sp, ps in zip(args.spacing, header.get_pixel_spacing(grid_header))]
    else:
        grid_spacing = args.spacing
        
    # paint the grid into the empty image volume
    for dim in range(grid_image.ndim):
        if 0 == grid_spacing[dim]: continue # skip dimension of 0 grid spacing supplied
        for offset in range(0, grid_image.shape[dim], grid_spacing[dim]):
            slicer = [slice(None)] * grid_image.ndim
            slicer[dim] = slice(offset, offset + 1)
            grid_image[slicer] = True
            
    # saving resulting grid volume
    save(grid_image, args.output, grid_header, args.force)
    

def list_of_integers_or_int(string, separator=','):
    if string.isdigit():
        return int(string)
    return list_of_integers(string, separator)

def list_of_integers(string, separator=','):
    values = string.split(separator)
    if not scipy.all(list(map(str.isdigit, values))):
        raise argparse.ArgumentTypeError('{} is not a "{}" separated list of integers'.format(string, separator))
    return list(map(int, values))

def list_of_floats(string, separator=','):
    values = string.split(separator)
    try:
        return list(map(float, values))
    except ValueError:
        raise argparse.ArgumentTypeError('{} is not a "{}" separated list of floats'.format(string, separator))

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    # get the number of dimensions in the image
    if args.example:
        args.example_image, args.example_header = load(args.example)
        dimensions = args.example_image.ndim
    else:
        dimensions = len(args.shape)
    
    # check and, if required, modify the spacing argument
    if isinstance(args.spacing, int):
        args.spacing = [args.spacing] * dimensions
    elif len(args.spacing) != dimensions:
        raise argparse.ArgumentTypeError('the grid spacing ({}) must contain the same number of elements as the output image has dimensions ({})'.format(','.join(map(str, args.spacing)), dimensions))
    
    # check further arguments
    if args.offset and len(args.offset) != dimensions:
        raise argparse.ArgumentTypeError('the offset ({}) must contain the same number of elements as the output image has dimensions ({})'.format(','.join(map(str, args.offset)), dimensions))
    if args.pixelspacing and len(args.pixelspacing) != dimensions:
        raise argparse.ArgumentTypeError('the supplied pixel spacing ({}) must contain the same number of elements as the output image has dimensions ({})'.format(','.join(map(str, args.pixelspacing)), dimensions))
    
    return args

def getParser():
    "Creates and returns the argparse parser object."
    # text
    epilog ="""
examples:
  %(prog)s -e example.nii grid.nii 10
      Generates an empty image with the same attributes as example.nii, overlays it
      with a regular grid of width 10 voxels and saves it as grid.nii.
  %(prog)s -e example.nii grid.nii 10,11,12 -r
      Same as above, but with an irregular grid and using real world coordinates
      (i.e. taking the voxel spacing of the image into account).
  %(prog)s -s 100,200 grid.nii 10,2 -p 0.5,3 
      Generates a 10x2 spaced grid in a 100x200 image with a voxel spacing of 0.5x3.
  %(prog)s -s 100,100,50 grid.nii 5,5,0 
      Generates a 100x100x50 3D volume but fills it only with a regular 5x5 2D grid
      over the first two dimensions.  
"""
    
    # command line argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, 
                                     description=__description__, epilog=epilog)
    parser.add_argument('output', help='Generated grid volume.')
    parser.add_argument('spacing', type=list_of_integers_or_int, help='The grid spacing. Can be a single digit for regular spacing in all dimensions or a colon-separated list of N integers, where N is the number of dimension in the generated volume. To skip the grid in one dimension, simply supply a 0 for it.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-e', '--example', dest='example', help='Option 1/2: Supply an image to create the grid volume by example (i.e. with same shape, voxel spacing and offset).')
    group.add_argument('-s', '--shape', type=list_of_integers, dest='shape', help='Option 2/2: Supply a colon-separated list of integers that constitute the target volumes shape.')
    
    parser.add_argument('-p', '--pixel-spacing', type=list_of_floats, dest='pixelspacing', help='Set the pixel spacing of the target volume by supplying a colon-separated list of N numbers, where N is the number of dimension in the generated volume.')
    parser.add_argument('-o', '--offset', type=list_of_floats, dest='offset', help='Set offset of the target volume by supplying a colon-separated list of N numbers, where N is the number of dimension in the generated volume.')

    parser.add_argument('-r', '--real', dest='real', action='store_true', help='Spacing is given in real world coordinates, rather than voxels. For this to make a difference, either the -e switch or the -p switch must be set.')
    
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='Silently override existing output images.')
    return parser 

if __name__ == "__main__":
    main()
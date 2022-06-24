#!/usr/bin/python3

"""
Takes a dicom folder, loads the contained slices and saves them as a proper 4D volume.

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

# third-party modules
import scipy

# path changes

# own modules
from medpy.core import Logger
from medpy.io import load, save
from medpy.core.exceptions import ArgumentError


# information
__author__ = "Oskar Maier"
__version__ = "d0.2.0, 2012-05-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Development"
__description__ = """
                  Takes a dicom folder, loads the contained slices and saves them as a proper 4D volume.
                  The supplied target dimension parameter determines the dimension along which to split the
                  original image and the consecutive slices parameter determines the offset after which to
                  split.
                
                  A typical use-case are DICOM images, which often come with the temporal and third spatial
                  dimension stacked on top of each other.
                  Let us assume a (5000, 200, 190) 3D image. In reality this file contains a number of 50
                  volume of 100x200x190, which each represent a point in time. More concretely, always 50
                  slices of the first dimension show the transformation of a 2D image in time. Then occurs
                  a visible jump, when the view changes in space from the 50th to the 51th slice. The
                  following 50 slices are the temporal transformation of this new spatial slice and then
                  occur another jump, and so on. 
                
                  Calling this script with a target dimension of 0 (meaning the first dimension of the
                  image containing the 5000 slices) and a consecutive slices parameter of 50 (which is used
                  to tell how many consecutive slices belong together), will result in a 4D image of the
                  shape (100, 50, 200, 190) containing the spatial volumes separated by an additional time
                  dimension.
                  
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
    
    data_3d, _ = load(args.input)
    
    # check parameters
    if args.dimension >= data_3d.ndim or args.dimension < 0:
        raise ArgumentError('The image has only {} dimensions. The supplied target dimension {} exceeds this number.'.format(
                    data_3d.ndim,
                    args.dimension))
    if not 0 == data_3d.shape[args.dimension] % args.offset:
        raise ArgumentError('The number of slices {} in the target dimension {} of the image shape {} is not dividable by the supplied number of consecutive slices {}.'.format(
                    data_3d.shape[args.dimension],
                    args.dimension,
                    data_3d.shape,
                    args.offset))
    
    # prepare empty target volume
    volumes_3d = data_3d.shape[args.dimension] / args.offset
    shape_4d = list(data_3d.shape)
    shape_4d[args.dimension] = volumes_3d
    data_4d = scipy.zeros([args.offset] + shape_4d, dtype=data_3d.dtype)
    
    logger.debug('Separating {} slices into {} 3D volumes of thickness {}.'.format(data_3d.shape[args.dimension], volumes_3d, args.offset))
        
    # iterate over 3D image and create sub volumes which are then added to the 4d volume
    for idx in range(args.offset):
        # collect the slices
        for sl in range(volumes_3d):
            idx_from = [slice(None), slice(None), slice(None)]
            idx_from[args.dimension] = slice(idx + sl * args.offset, idx + sl * args.offset + 1)
            idx_to = [slice(None), slice(None), slice(None)]
            idx_to[args.dimension] = slice(sl, sl+1)
            #print 'Slice {} to {}.'.format(idx_from, idx_to)
            data_4d[idx][idx_to] = data_3d[idx_from]
        
    # flip dimensions such that the newly created is the last
    data_4d = scipy.swapaxes(data_4d, 0, 3)
        
    # save resulting 4D volume
    save(data_4d, args.output, False, args.force)
    
    logger.info("Successfully terminated.")

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', help='Source directory.')
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('dimension', type=int, help='The dimension in which to perform the cut (starting from 0).')
    parser.add_argument('offset', type=int, help='How many consecutive slices belong together before a shift occurs. / The offset between the volumes.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()

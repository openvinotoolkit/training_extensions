#!/usr/bin/python3

"""
Reslices 4D information stacked into a 3D volume into a real 4D image.

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
from medpy.io import load, save
from medpy.core.exceptions import ArgumentError


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2012-05-25"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Takes a a 3D volume containing 4D information and converts it into a
                  real 4D image.
                  This is achieved by taking every offset slice, starting from the first,
                  of the input 4D volume and then by combining them into a 3D volume.
                  Then repeats the process starting from the second slice, etc.
                  The new dimension will be appended to the already existing once.
                  
                  A typical use case are dicom images. These often come with the time
                  dimension represented by stacking various 3D volumes on top of each
                  other in one of the spatial dimensions. These can be converted in
                  proper 4D volumes with this script.
                  
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
    
    # load 3d image
    data_3d, header_3d = load(args.input)
    
    # check if supplied dimension parameter is inside the images dimensions
    if args.dimension >= data_3d.ndim or args.dimension < 0:
        raise ArgumentError('The supplied cut-dimension {} exceeds the number of input volume dimensions {}.'.format(args.dimension, data_3d.ndim))
    
    # check if the supplied offset parameter is a divider of the cut-dimensions slice number
    if not 0 == data_3d.shape[args.dimension] % args.offset:
        raise ArgumentError('The offset is not a divider of the number of slices in cut dimension ({} / {}).'.format(data_3d.shape[args.dimension], args.offset))
        
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
    data_4d = scipy.swapaxes(data_4d, 0, args.dimension + 1)
    data_4d = scipy.rollaxis(data_4d, 0, 4)
        
    # save resulting 4D volume
    save(data_4d, args.output, header_3d, args.force)
    
    logger.info("Successfully terminated.")

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='Source volume.')
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('dimension', type=int, help='The dimension in which to perform the cut (starting from 0).')
    parser.add_argument('offset', type=int, help='The offset between the slices.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    return parser    

if __name__ == "__main__":
    main()

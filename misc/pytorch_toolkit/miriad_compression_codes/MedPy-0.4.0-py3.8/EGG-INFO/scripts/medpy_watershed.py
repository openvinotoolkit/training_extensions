#!/usr/bin/python3

"""
Executes the watershed algorithm over images.
requires the skimage package to be installed.

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
import numpy
from scipy.ndimage.measurements import label
from skimage.morphology import watershed

# path changes

# own modules
from medpy.io import load, save
from medpy.core import Logger, ArgumentError
from medpy.filter import local_minima


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2013-12-11"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Applies the watershed segmentation an image using the supplied
                  parameters.
                  Note that this version does not take the voxel-spacing into account.
                                  
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
    
    # check if output image exists (will also be performed before saving, but as the watershed might be very time intensity, a initial check can save frustration)
    if not args.force:
        if os.path.exists(args.output):
            raise ArgumentError('The output image {} already exists.'.format(args.output))
    
    # loading images
    data_input, header_input = load(args.input)
    if args.mask:
        mask = load(args.mask)[0].astype(numpy.bool)
    else:
        mask = None
    
    # extract local minima and convert to markers
    logger.info('Extract local minima with minimum distance of {}...'.format(args.mindist))
    lm, _ = local_minima(data_input, args.mindist)
    lm_indices = tuple([numpy.asarray(x) for x in lm.T])
    minima_labels = numpy.zeros(data_input.shape, dtype=numpy.uint64)
    minima_labels[lm_indices] = 1
    if not None == mask:
        minima_labels[~mask] = 0
    minima_labels, _ = label(minima_labels)
    
    # apply the watershed
    logger.info('Watershedding...')
    data_output = watershed(data_input, minima_labels, mask=mask)

    # save file
    save(data_output, args.output, header_input, args.force)
    
    logger.info('Successfully terminated.')

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='Source volume (usually a gradient image).')
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('--mindist', type=int, default=2, help='The minimum distance between local minima in voxel units.')
    parser.add_argument('--mask', help='Optional binary mask image denoting the area over which to compute the watershed.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    
    return parser
    
if __name__ == "__main__":
    main()

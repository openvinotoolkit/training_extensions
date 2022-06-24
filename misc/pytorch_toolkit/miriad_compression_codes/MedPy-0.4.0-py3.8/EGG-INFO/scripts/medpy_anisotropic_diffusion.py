#!/usr/bin/python3

"""
Executes gradient anisotropic diffusion filter over an image.

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
from medpy.io import load, save, get_pixel_spacing
from medpy.core import Logger
from medpy.filter.smoothing import anisotropic_diffusion


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.0, 2013-08-24"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Executes gradient anisotropic diffusion filter over an image.
                  This smoothing algorithm is edges preserving.
                  
                  Note that the images voxel-spacing will be taken into account.
                                  
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
        if os.path.exists(args.output):
            raise parser.error('The output image {} already exists.'.format(args.output))
    
    # loading image
    data_input, header_input = load(args.input)
    
    # apply the watershed
    logger.info('Applying anisotropic diffusion with settings: niter={} / kappa={} / gamma={}...'.format(args.iterations, args.kappa, args.gamma))
    data_output = anisotropic_diffusion(data_input, args.iterations, args.kappa, args.gamma, get_pixel_spacing(header_input))

    # save file
    save(data_output, args.output, header_input, args.force)
    
    logger.info('Successfully terminated.')

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('input', help='Source volume.')
    parser.add_argument('output', help='Target volume.')
    parser.add_argument('-i', '--iterations', type=int, default=1, help='The number of smoothing iterations. Strong parameter.')
    parser.add_argument('-k', '--kappa', type=int, default=50, help='The algorithms kappa parameter. The higher the more edges are smoothed over.')
    parser.add_argument('-g', '--gamma', type=float, default=0.1, help='The algorithms gamma parameter. The higher, the stronger the plateaus between edges are smeared.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', dest='force', action='store_true', help='Silently override existing output images.')
    
    return parser
    
if __name__ == "__main__":
    main()

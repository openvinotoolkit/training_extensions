#!/usr/bin/python3

"""
Execute a graph cut on a voxel image based on some foreground and background markers.

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
from argparse import RawTextHelpFormatter
import argparse
import logging
import os

# third-party modules
import scipy

# path changes

# own modules
from medpy.core import ArgumentError, Logger
from medpy.io import load, save, header
from medpy import graphcut
from medpy.graphcut.wrapper import split_marker



# information
__author__ = "Oskar Maier"
__version__ = "r0.3.1, 2012-03-23"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
                  Perform a binary graph cut using Boykov's max-flow/min-cut algorithm.
                  
                  This implementation does only compute a boundary term and does not use
                  any regional term. The desired boundary term can be selected via the
                  --boundary argument. Depending on the selected term, an additional
                  image has to be supplied as badditional.
                  
                  In the case of the difference of means, it is the original image.
                  
                  Furthermore the algorithm requires a binary image with foreground
                  markers and a binary image with background markers.
                  
                  Additionally a filename for the created binary mask marking foreground
                  and background has to be supplied.
                  
                  Note that the input images must be of the same dimensionality,
                  otherwise an exception is thrown.
                  Note to take into account the input images orientation.
                  Note that the quality of the resulting segmentations depends also on
                  the quality of the supplied markers.
                  
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
        
    # check if output image exists
    if not args.force:
        if os.path.exists(args.output):
            logger.warning('The output image {} already exists. Exiting.'.format(args.output))
            exit(-1)
            
    # select boundary term
    ['diff_linear', 'diff_exp', 'diff_div', 'diff_pow', 'max_linear', 'max_exp', 'max_div', 'max_pow']
    if 'diff_linear' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_difference_linear
        logger.info('Selected boundary term: linear difference of intensities')
    elif 'diff_exp' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_difference_exponential
        logger.info('Selected boundary term: exponential difference of intensities')
    elif 'diff_div' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_difference_division
        logger.info('Selected boundary term: divided difference of intensities')
    elif 'diff_pow' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_difference_power
        logger.info('Selected boundary term: power based / raised difference of intensities')
    elif 'max_linear' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_maximum_linear
        logger.info('Selected boundary term: linear maximum of intensities')
    elif 'max_exp' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_maximum_exponential
        logger.info('Selected boundary term: exponential maximum of intensities')
    elif 'max_div' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_maximum_division
        logger.info('Selected boundary term: divided maximum of intensities')
    elif 'max_pow' == args.boundary:
        boundary_term = graphcut.energy_voxel.boundary_maximum_power
        logger.info('Selected boundary term: power based / raised maximum of intensities')

    # load input images
    badditional_image_data, reference_header = load(args.badditional)
    markers_image_data, _ = load(args.markers)
    
    # split marker image into fg and bg images
    fgmarkers_image_data, bgmarkers_image_data = split_marker(markers_image_data)
       
    # check if all images dimensions are the same
    if not (badditional_image_data.shape == fgmarkers_image_data.shape == bgmarkers_image_data.shape):
        logger.critical('Not all of the supplied images are of the same shape.')
        raise ArgumentError('Not all of the supplied images are of the same shape.')

    # extract spacing if required
    if args.spacing:
        spacing = header.get_pixel_spacing(reference_header)
        logger.info('Taking spacing of {} into account.'.format(spacing))
    else:
        spacing = False

    # generate graph
    logger.info('Preparing BK_MFMC C++ graph...')
    gcgraph = graphcut.graph_from_voxels(fgmarkers_image_data,
                                         bgmarkers_image_data,
                                         boundary_term = boundary_term,
                                         boundary_term_args = (badditional_image_data, args.sigma, spacing))
    
    # execute min-cut
    logger.info('Executing min-cut...')
    maxflow = gcgraph.maxflow()
    logger.debug('Maxflow is {}'.format(maxflow))
    
    # reshape results to form a valid mask
    logger.info('Applying results...')
    result_image_data = scipy.zeros(bgmarkers_image_data.size, dtype=scipy.bool_)
    for idx in range(len(result_image_data)):
        result_image_data[idx] = 0 if gcgraph.termtype.SINK == gcgraph.what_segment(idx) else 1    
    result_image_data = result_image_data.reshape(bgmarkers_image_data.shape)
    
    # save resulting mask    
    save(result_image_data.astype(scipy.bool_), args.output, reference_header, args.force)

    logger.info('Successfully terminated.')

def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    return parser.parse_args()

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('sigma', type=float, help='The sigma required for the boundary terms.')
    parser.add_argument('badditional', help='The additional image required by the boundary term. See there for details.')
    parser.add_argument('markers', help='Image containing the foreground (=1) and background (=2) markers.')
    parser.add_argument('output', help='The output image containing the segmentation.')
    parser.add_argument('--boundary', default='diff_exp', help='The boundary term to use. Note that the ones prefixed with diff_ require the original image, while the ones prefixed with max_ require the gradient image.', choices=['diff_linear', 'diff_exp', 'diff_div', 'diff_pow', 'max_linear', 'max_exp', 'max_div', 'max_pow'])
    parser.add_argument('-s', dest='spacing', action='store_true', help='Set this flag to take the pixel spacing of the image into account. The spacing data will be extracted from the baddtional image.')
    parser.add_argument('-f', dest='force', action='store_true', help='Set this flag to silently override files that exist.')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Display more information.')
    parser.add_argument('-d', dest='debug', action='store_true', help='Display debug information.')
    
    return parser    

if __name__ == "__main__":
    main()
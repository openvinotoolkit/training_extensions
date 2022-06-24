#!/usr/bin/python3

"""
Standardizes the intensity range / profile of a number of similar images.

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
import pickle
import argparse
import logging

# third-party modules
import numpy

# path changes

# own modules
from medpy.core import Logger
from medpy.core.exceptions import ArgumentError
from medpy.io import load, save
from medpy.utilities.argparseu import sequenceOfIntegersGeAscendingStrict
from medpy.filter import IntensityRangeStandardization


# information
__author__ = "Oskar Maier"
__version__ = "r0.1.1, 2013-10-11"
__email__ = "oskar.maier@googlemail.com"
__status__ = "Release"
__description__ = """
Standardizes the intensity range / profile of a number of similar images.

Takes a number of images that display the same scene (most commonly MRI volumes of the
same body region) and learns an average intensity range model from these. This model can
then be used to transfer the training image set and other, formerly unseen images, to the
learned average intensity range. Such prepared, these images display the same intensity 
profiles for the same structures.

The employed algorithm guarantees a lossless intensity transformation and throws an
exception, should the model require re-training.

The application requires the supplied images to be stripped of their background. This can
either be done by setting a threshold value or by supplying a foreground-mask for each
image.

The script provide two mutually exclusive functionalities:
    (1) the training of a new model and an optional application to the training images
    (2) the application of a stored model to a number of images
Depending on the application, different arguments are require that are reflected by the
argument grouping.

The implementation is based on:
[1] Nyul, L.G.; Udupa, J.K.; Xuan Zhang, "New variants of a method of MRI scale
    standardization," Medical Imaging, IEEE Transactions on , vol.19, no.2, pp.143-150,
    Feb. 2000
    
For more details on the algorithm, see the medpy.filter.IntensityRangeStandardization class.

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
    
    # loading input images (as image, header pairs)
    images = []
    headers = []
    for image_name in args.images:
        i, h = load(image_name)
        images.append(i)
        headers.append(h)
    
    # loading binary foreground masks if supplied, else create masks from threshold value
    if args.masks:
        masks = [load(mask_name)[0].astype(numpy.bool) for mask_name in args.masks]
    else:
        masks = [i > args.threshold for i in images]
    
    # if in application mode, load the supplied model and apply it to the images
    if args.lmodel:
        logger.info('Loading the model and transforming images...')
        with open(args.lmodel, 'r') as f:
            trained_model = pickle.load(f)
            if not isinstance(trained_model, IntensityRangeStandardization):
                raise ArgumentError('{} does not seem to be a valid pickled instance of an IntensityRangeStandardization object'.format(args.lmodel))
            transformed_images = [trained_model.transform(i[m], surpress_mapping_check = args.ignore) for i, m in zip(images, masks)]
            
    # in in training mode, train the model, apply it to the images and save it
    else:
        logger.info('Training the average intensity model...')
        irs = IntensityRangeStandardization()
        trained_model, transformed_images = irs.train_transform([i[m] for i, m in zip(images, masks)], surpress_mapping_check = args.ignore)
        logger.info('Saving the trained model as {}...'.format(args.smodel))
        with open(args.smodel, 'wb') as f:
                pickle.dump(trained_model, f)
                
    # save the transformed images
    if args.simages:
        logger.info('Saving intensity transformed images to {}...'.format(args.simages))
        for ti, i, m, h, image_name in zip(transformed_images, images, masks, headers, args.images):
            i[m] = ti
            save(i, '{}/{}'.format(args.simages, image_name.split('/')[-1]), h, args.force)
    
    logger.info('Terminated.')
    

    
def getArguments(parser):
    "Provides additional validation of the arguments collected by argparse."
    args = parser.parse_args()
    
    # check mutual exlusive and reaquired arguments
    if args.lmodel and args.smodel:
        parser.error('only one of --load-model and --save-model can be supplied, as they decide on whether to apply the application or the training mode')
    if not args.lmodel and not args.smodel:
        parser.error('exactly one of --load-model or --save-model has to be supplied')
    
    # application mode
    if args.lmodel:
        if not os.path.isfile(args.lmodel):
            parser.error('the supplied model file {} does not exist'.format(args.lmodel))
        if not args.simages:
            parser.error('--save-images must be supplied when running the application mode')
    
    # training mode
    if args.smodel:
        if not args.landmarkp in ('L2', 'L3', 'L4'):
            args.landmarkp = sequenceOfIntegersGeAscendingStrict(args.landmarkp)
        if not 'auto' == args.stdspace:
            args.stdspace = sequenceOfIntegersGeAscendingStrict(args.stdspace)
        if not args.force and os.path.isfile(args.smodel):
            parser.error('the target model file {} already exists'.format(args.smodel))
        
    # others
    if args.simages:
        if not os.path.isdir(args.simages):
            parser.error('--save-images must be a valid directory')
    if args.masks and len(args.masks) != len(args.images):
        parser.error('the same number of masks must be passed to --masks as images have been supplied') 
    
    return args

def getParser():
    "Creates and returns the argparse parser object."
    parser = argparse.ArgumentParser(description=__description__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('images', nargs='+', help='The images used for training (in the learning case) or to transform (in the transformation case)')

    apply_group = parser.add_argument_group('apply an existing model')
    apply_group.add_argument('--load-model', dest='lmodel', default=False, help='Location of the pickled intensity range model to load. Activated application mode.')
    
    train_group = parser.add_argument_group('train a new model and save and/or apply it')
    train_group.add_argument('--save-model', dest='smodel', default=False, help='Save the trained model under this name as a pickled object (should end in .pkl). Activates training mode.')
    train_group.add_argument('--cutoffp', dest='cutoffp', type=sequenceOfIntegersGeAscendingStrict, default='1,99', help='Colon-separated lower and upper cut-off percentile values to exclude intensity outliers during the model training.')
    train_group.add_argument('--landmarkp', dest='landmarkp', default='L4', help='The landmark percentiles, based on which to train the model. Can be L2, L3, L4 or a colon-separated, ordered list of percentiles.')
    train_group.add_argument('--stdspace', dest='stdspace', default='auto', help='Two colon-separated intensity values to roughly define the average intensity space to learn. In most cases should be left set to \'auto\'')
    
    shared_group = parser.add_argument_group('shared arguments')
    shared_group.add_argument('--save-images', dest='simages', default=False, help='Save the transformed images under this location. Required for the application mode, optional for the learning mode.')
    shared_group.add_argument('--threshold', type=float, default=0, help='All voxel with an intensity > threshold are considered as foreground. Supply either this or a mask for each image.')
    shared_group.add_argument('--masks', nargs='+', help='A number of binary foreground mask, one for each image. Alternative to supplying a threshold. Overrides the threshold parameter if supplied.')
    shared_group.add_argument('--ignore', dest='ignore', action='store_true', help='Ignore possible loss of information during the intensity transformation. Should only be used when you know what you are doing.')
    
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Verbose output')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Display debug information.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='Overwrite existing files (both model and images)')
    return parser
    
if __name__ == "__main__":
    main()        
# Copyright (C) 2013 Oskar Maier
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author Oskar Maier
# version r0.3.1
# since 2012-05-28
# status Release

# build-in modules
import os.path

# third-party modules
import numpy as np
import SimpleITK as sitk

# own modules
from .header import Header
from ..core import Logger
from ..core import ImageLoadingError

# code
def load(image):
    r"""
    Loads the ``image`` and returns a ndarray with the image's pixel content as well as
    a header object.
    
    The header can, with restrictions, be used to extract additional meta-information
    about the image (e.g. using the methods in `~medpy.io.Header`). Additionally
    it serves as meta-data container that can be passes to `~medpy.io.save.save` when the
    altered image is saved to the hard drive again. Note that the transfer of meta-data is
    only possible, and even then not guaranteed, when the source and target image formats
    are the same.
    
    MedPy relies on SimpleITK, which enables the power of ITK for image loading and saving.
    The supported image file formats should include at least the following.

    Medical formats:

    - ITK MetaImage (.mha/.raw, .mhd)
    - Neuroimaging Informatics Technology Initiative (NIfTI) (.nia, .nii, .nii.gz, .hdr, .img, .img.gz)
    - Analyze (plain, SPM99, SPM2) (.hdr/.img, .img.gz)
    - Digital Imaging and Communications in Medicine (DICOM) (.dcm, .dicom)
    - Digital Imaging and Communications in Medicine (DICOM) series (<directory>/)
    - Nearly Raw Raster Data (Nrrd) (.nrrd, .nhdr) 
    - Medical Imaging NetCDF (MINC) (.mnc, .MNC)
    - Guys Image Processing Lab (GIPL) (.gipl, .gipl.gz)

    Microscopy formats:

    - Medical Research Council (MRC) (.mrc, .rec)
    - Bio-Rad (.pic, .PIC)
    - LSM (Zeiss) microscopy images (.tif, .TIF, .tiff, .TIFF, .lsm, .LSM)
    - Stimulate / Signal Data (SDT) (.sdt)

    Visualization formats:

    - VTK images (.vtk)

    Other formats:
    
    - Portable Network Graphics (PNG) (.png, .PNG)
    - Joint Photographic Experts Group (JPEG) (.jpg, .JPG, .jpeg, .JPEG)
    - Tagged Image File Format (TIFF) (.tif, .TIF, .tiff, .TIFF)
    - Windows bitmap (.bmp, .BMP)
    - Hierarchical Data Format (HDF5) (.h5 , .hdf5 , .he5)
    - MSX-DOS Screen-x (.ge4, .ge5)
        
    For informations about which image formats, dimensionalities and pixel data types
    your current configuration supports, run `python3 tests/support.py > myformats.log`.

    Further information see https://simpleitk.readthedocs.io .
    
    Parameters
    ----------
    image : string
        Path to the image to load.
    
    Returns
    -------
    image_data : ndarray
        The image data as numpy array with order `x,y,z,c`.
    image_header : Header
        The image metadata as :mod:`medpy.io.Header`.
    
    Raises
    ------
    ImageLoadingError
        If the image could not be loaded due to some reason.
    """
    logger = Logger.getInstance()
    logger.info('Loading image {}...'.format(image))

    if not os.path.exists(image):
        raise ImageLoadingError('The supplied image {} does not exist.'.format(image))

    if os.path.isdir(image):
        # !TODO: this does not load the meta-data, find a way to load it from a series, too
        logger.info('Loading image as DICOM series. If more than one found in folder {} defaulting to first.'.format(image))
        sitkimage = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(image))
    else:
        sitkimage = sitk.ReadImage(image)
        
    # Make image array data and header
    header = Header(sitkimage=sitkimage)
    image = sitk.GetArrayFromImage(sitkimage)

    # Roll axes from z,y,x,c to x,y,z,c
    if image.ndim == 4:
        image = np.moveaxis(image, -1, 0)
    image = image.T

    return image, header

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
# version r0.2.1
# since 2012-05-28
# status Release

# build-in modules
import os

# third-party modules
import numpy as np
import SimpleITK as sitk

# own modules
from ..core import Logger
from ..core import ImageSavingError

# code
def save(arr, filename, hdr = False, force = True, use_compression = False):
    r"""
    Save the image ``arr`` as filename using information encoded in ``hdr``. The target image
    format is determined by the ``filename`` suffix. If the ``force`` parameter is set to true,
    an already existing image is overwritten silently. Otherwise an error is thrown.
    
    The header (``hdr``) object is the one returned by `~medpy.io.load.load` and is used
    opportunistically, possibly loosing some meta-information.
    
    Generally this function does not guarantee, that metadata other than the image shape
    and pixel data type are kept.
    
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
    arr : array_like
        The image data with order `x,y,z,c`.
    filename : string
        Where to save the image; path and filename including the image suffix.
    hdr : object
        The image header containing the metadata.
    force : bool
        Set to True to overwrite already exiting image silently.
    use_compression : bool
        Use data compression of the target format supports it.
    
    Raises
    ------
    ImageSavingError
        If the image could not be saved due to various reasons
    """
    logger = Logger.getInstance()
    logger.info('Saving image as {}...'.format(filename))
    
    # Check image file existance
    if not force and os.path.exists(filename):
        raise ImageSavingError('The target file {} already exists.'.format(filename))
    
    # Roll axes from x,y,z,c to z,y,x,c
    if arr.ndim == 4:
        arr = np.moveaxis(arr, -1, 0)
    arr = arr.T

    sitkimage = sitk.GetImageFromArray(arr)
  
    # Copy met-data as far as possible
    if hdr:
        hdr.copy_to(sitkimage)
        
    sitk.WriteImage(sitkimage, filename, use_compression)

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
# version r0.2.3
# since 2012-06-01
# status Release

# build-in modules
import warnings

# third-party modules
import numpy as np

# own modules
from ..core import Logger

# code
def get_voxel_spacing(hdr):
    r"""
    Extracts the voxel spacing from an image header.

    Notes
    -----
    It is recommended to call `hdr.get_voxel_spacing()` instead
    of this function.
    
    Parameters
    ----------
    hdr : medpy.io.Header
        An image header as returned by `load`.
    
    Returns
    -------
    pixel_spacing : tuple of floats
        The image's pixel spacing.
    """
    return hdr.get_voxel_spacing()

def get_pixel_spacing(hdr):
    r"""Depreciated synonym of `~medpy.io.header.get_voxel_spacing`."""
    warnings.warn('get_pixel_spacing() is depreciated, use set_voxel_spacing() instead', category=DeprecationWarning)
    return get_voxel_spacing(hdr)

def get_offset(hdr):
    r"""
    Extracts the image offset (akak origin) from an image header.

    Notes
    -----
    It is recommended to call `hdr.get_offset()` instead
    of this function.
    It can be assumed that the offset is measured from the center point of
    the first pixel, which SimpleITK promises independent of the file format.
    Some formats do not specify a header field for the offset, thus zeros
    are returned.
    
    Parameters
    ----------
    hdr : medpy.io.Header
        An image header as returned by `load`.
    
    Returns
    -------
    offset : tuple of floats
        The image's offset.
    """
    return hdr.get_offset()

def set_voxel_spacing(hdr, spacing):
    r"""
    Sets the voxel spacing in an image header.
    
    Notes
    -----
    It is recommended to call `hdr.set_voxel_spacing()` instead
    of this function.    

    Parameters
    ----------
    hdr : medpy.io.Header
        An image header as returned by `load`.
    pixel_spacing : tuple of floats
        The desired pixel spacing.
    """
    hdr.set_voxel_spacing(spacing)

def set_pixel_spacing(hdr, spacing):
    r"""Depreciated synonym of `~medpy.io.header.set_voxel_spacing`."""
    warnings.warn('get_pixel_spacing() is depreciated, use set_voxel_spacing() instead', category=DeprecationWarning)
    set_voxel_spacing(hdr, spacing)
   
def set_offset(hdr, offset):
    r"""
    Sets the offset (aka origin) in the image header.
    
    Notes
    -----
    It is recommended to call `hdr.set_offset()` instead
    of this function.    
    The offset is based on the center of the first voxel.
    See also `get_offset` for more details.

    Parameters
    ----------
    hdr : medpy.io.Header
        An image header as returned by `load`.
    offset : tuple of floats
        The desired offset.
    """
    hdr.set_offset(offset)


def copy_meta_data(hdr_to, hdr_from):
    r"""
    Copy image meta data (voxel spacing and offset) from one header to another.
    
    Parameters
    ----------
    hdr_to : object
        An image header as returned by `load`.
    hdr_from : object
        An image header as returned by `load`.
    """
    warnings.warn('copy_meta_data() is depreciated and may be removed in future versions', category=DeprecationWarning)
    logger = Logger.getInstance()
    try:
        set_pixel_spacing(hdr_to, get_pixel_spacing(hdr_from))
    except AttributeError as e:
        logger.warning('The voxel spacing could not be set correctly. Signaled error: {}'.format(e))
    try:
        set_offset(hdr_to, get_offset(hdr_from))
    except AttributeError as e:
        logger.warning('The image offset could not be set correctly. Signaled error: {}'.format(e))


class Header:
    r"""
    A medpy header object.

    Stores spacing, offset/origin, direction, and possibly further meta information.
    Provide at least one of the parameters. Missing information is extracted from
    the ``sitkimage`` or, if not supplied, set to a default value. 

    Parameters
    ----------
    spacing : tuple of floats
        the image's voxel spacing
        defaults to a tuple of `1.0`s
    offset : tuple of floats
        the image's offset/origin
        defaults to a tuple of `0.0`s
    direction : ndarray
        the image's affine transformation matrix
        must be of square shape
        default to the identity matrix
    sitkimage : sitk.Image
        the simple itk image as loaded
    """

    def __init__(self, spacing=None, offset=None, direction=None, sitkimage=None):
        assert \
            sitkimage is not None or \
            spacing is not None or \
            offset is not None or \
            direction is not None

        # determin the image's ndim and default data types
        if direction is not None:
            direction = np.asarray(direction)
            ndim = len(direction.shape[0])
        elif offset is not None:
            offset = tuple(offset)
            ndim = len(offset)
        elif spacing is not None:
            spacing = tuple(spacing)
            ndim = len(spacing)
        else:
            ndim = len(sitkimage.GetSpacing())
        
        # set missing information to extracted or default values
        if spacing is None:
            spacing = sitkimage.GetSpacing() if sitkimage is not None else (1.0, ) * ndim
        if offset is None:
            offset = sitkimage.GetOrigin() if sitkimage is not None else (0.0, ) * ndim
        if direction is None:
            direction = np.asarray(sitkimage.GetDirection()).reshape(ndim, ndim) if sitkimage is not None else np.identity(ndim)
            
        # assert consistency
        assert len(spacing) == len(offset)
        assert direction.ndim == 2
        assert len(spacing) == direction.shape[0]
        assert direction.shape[0] == direction.shape[1]

        # set members
        self.spacing = spacing
        self.offset = offset
        self.direction = direction
        self.sitkimage = sitkimage

    def copy_to(self, sitkimage):
        """
        Copy all stored meta information info to an sitk Image.

        Note that only the spacing and the offset/origin information
        are guaranteed to be preserved, although the method also
        tries to copy other meta information such as DICOM tags.

        Parameters
        ----------
        sitkimage : sitk.Image
            the sitk Image object to which to copy the information

        Returns
        -------
        sitkimage : sitk.Image
            the passed sitk Image object
        """
        if self.sitkimage is not None:
            for k in self.sitkimage.GetMetaDataKeys():
                sitkimage.SetMetaData(k, self.sitkimage.GetMetaData(k))

        ndim = len(sitkimage.GetSize())
        spacing, offset, direction = self.get_info_consistent(ndim)
            
        sitkimage.SetSpacing(spacing)
        sitkimage.SetOrigin(offset)
        sitkimage.SetDirection(tuple(direction.flatten()))
        
        return sitkimage
        
    def get_info_consistent(self, ndim):
        """
        Returns the main meta-data information adapted to the supplied
        image dimensionality.

        It will try to resolve inconsistencies and other conflicts,
        altering the information avilable int he most plausible way.

        Parameters
        ----------
        ndim : int
            image's dimensionality
        
        Returns
        -------
        spacing : tuple of floats
        offset : tuple of floats
        direction : ndarray
        """
        if ndim > len(self.spacing):
            spacing = self.spacing + (1.0, ) * (ndim - len(self.spacing))
        else:
            spacing = self.spacing[:ndim]

        if ndim > len(self.offset):
            offset = self.offset + (0.0, ) * (ndim - len(self.offset))
        else:
            offset = self.offset[:ndim]

        if ndim > self.direction.shape[0]:
            direction = np.identity(ndim)
            direction[:self.direction.shape[0], :self.direction.shape[0]] = self.direction
        else:
            direction = self.direction[:ndim, :ndim]
        
        return spacing, offset, direction

    def set_voxel_spacing(self, spacing):
        """
        Set image's spacing.

        Parameters
        ----------
        spacing : tuple of floats
            the new image voxel spacing
            take care that image and spacing dimensionalities match
        """        
        self.spacing = tuple(spacing)
    
    def set_offset(self, offset):
        """
        Set image's offset.

        Parameters
        ----------
        offset : tuple of floats
            the new image offset / origin
            take care that image and offset dimensionalities match
        """
        self.offset = tuple(offset)

    def set_direction(self, direction):
        """
        Set image's direction.

        Returns
        -------
        direction : tuple of floats
            the image's direction / affine transformation matrix
            must be of square shape
            default to the identity matrix
        """
        self.direction = np.asarray(direction)
    
    def get_voxel_spacing(self):
        """
        Get image's spacing.
        
        Returns
        -------
        spacing : tuple of floats
            the image's spacing
        """
        return self.spacing
    
    def get_offset(self):
        """
        Get image's offset.

        Returns
        -------
        offset : tuple of floats
            the image's offset / origin
        """
        return self.offset

    def get_direction(self):
        """
        Get image's direction.

        Returns
        -------
        direction : ndarray
            the image's direction / affine transformation matrix
            of square shape
        """
        return self.direction    

    def get_sitkimage(self):
        """
        Get underlying sitk Image object.
        
        Returns
        -------
        image-object : sitk.Image or None
            the underlying sitk image object if set
        """
        return self.sitkimage

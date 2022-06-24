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
# version r0.3.0
# since 2013-11-29
# status Release

# build-in modules
import itertools
import numbers
import math

# third-party modules
import numpy
from scipy.ndimage.filters import convolve, gaussian_filter, minimum_filter
from scipy.ndimage._ni_support import _get_output
from scipy.ndimage.interpolation import zoom

# own modules
from .utilities import pad, __make_footprint
from ..io import header

# code
def sls(minuend, subtrahend, metric = "ssd", noise = "global", signed = True,
        sn_size = None, sn_footprint = None, sn_mode = "reflect", sn_cval = 0.0,
        pn_size = None, pn_footprint = None, pn_mode = "reflect", pn_cval = 0.0):
    r"""
    Computes the signed local similarity between two images.

    Compares a patch around each voxel of the minuend array to a number of patches
    centered at the points of a search neighbourhood in the subtrahend. Thus, creates
    a multi-dimensional measure of patch similarity between the minuend and a
    corresponding search area in the subtrahend.

    This filter can also be used to compute local self-similarity, obtaining a
    descriptor similar to the one described in [1]_.

    Parameters
    ----------
    minuend : array_like
        Input array from which to subtract the subtrahend.
    subtrahend : array_like
        Input array to subtract from the minuend.
    metric : {'ssd', 'mi', 'nmi', 'ncc'}, optional
        The `metric` parameter determines the metric used to compute the
        filter output. Default is 'ssd'.
    noise : {'global', 'local'}, optional
        The `noise` parameter determines how the noise is handled. If set
        to 'global', the variance determining the noise is a scalar, if
        set to 'local', it is a Gaussian smoothed field of estimated local
        noise. Default is 'global'.
    signed : bool, optional
        Whether the filter output should be signed or not. If set to 'False',
        only the absolute values will be returned. Default is 'True'.
    sn_size : scalar or tuple, optional
        See sn_footprint, below
    sn_footprint : array, optional
        The search neighbourhood.
        Either `sn_size` or `sn_footprint` must be defined. `sn_size` gives
        the shape that is taken from the input array, at every element
        position, to define the input to the filter function.
        `sn_footprint` is a boolean array that specifies (implicitly) a
        shape, but also which of the elements within this shape will get
        passed to the filter function. Thus ``sn_size=(n,m)`` is equivalent
        to ``sn_footprint=np.ones((n,m))``. We adjust `sn_size` to the number
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and `sn_size` is 2, then the actual size used is
        (2,2,2).
    sn_mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `sn_mode` parameter determines how the array borders are
        handled, where `sn_cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    sn_cval : scalar, optional
        Value to fill past edges of input if `sn_mode` is 'constant'. Default
        is 0.0
    pn_size : scalar or tuple, optional
        See pn_footprint, below
    pn_footprint : array, optional
        The patch over which the distance measure is applied.
        Either `pn_size` or `pn_footprint` must be defined. `pn_size` gives
        the shape that is taken from the input array, at every element
        position, to define the input to the filter function.
        `pn_footprint` is a boolean array that specifies (implicitly) a
        shape, but also which of the elements within this shape will get
        passed to the filter function. Thus ``pn_size=(n,m)`` is equivalent
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and `pn_size` is 2, then the actual size used is
        (2,2,2).
    pn_mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `pn_mode` parameter determines how the array borders are
        handled, where `pn_cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    pn_cval : scalar, optional
        Value to fill past edges of input if `pn_mode` is 'constant'. Default
        is 0.0
        
    Returns
    -------
    sls : ndarray
        The signed local similarity image between subtrahend and minuend.

    References
    ----------
    
    .. [1] Mattias P. Heinrich, Mark Jenkinson, Manav Bhushan, Tahreema Matin, Fergus V. Gleeson, Sir Michael Brady, Julia A. Schnabel
           MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration
           Medical Image Analysis, Volume 16, Issue 7, October 2012, Pages 1423-1435, ISSN 1361-8415
           http://dx.doi.org/10.1016/j.media.2012.05.008
    """
    minuend = numpy.asarray(minuend)
    subtrahend = numpy.asarray(subtrahend)
    
    if numpy.iscomplexobj(minuend):
        raise TypeError('complex type not supported')
    if numpy.iscomplexobj(subtrahend):
        raise TypeError('complex type not supported')  
    
    mshape = [ii for ii in minuend.shape if ii > 0]
    sshape = [ii for ii in subtrahend.shape if ii > 0]
    if not len(mshape) == len(sshape):
        raise RuntimeError("minuend and subtrahend must be of same shape")
    if not numpy.all([sm == ss for sm, ss in zip(mshape, sshape)]):
        raise RuntimeError("minuend and subtrahend must be of same shape")
    
    sn_footprint = __make_footprint(minuend, sn_size, sn_footprint)
    sn_fshape = [ii for ii in sn_footprint.shape if ii > 0]
    if len(sn_fshape) != minuend.ndim:
        raise RuntimeError('search neighbourhood footprint array has incorrect shape.')
    
    #!TODO: Is this required?
    if not sn_footprint.flags.contiguous:
        sn_footprint = sn_footprint.copy()
    
    # created a padded copy of the subtrahend, whereas the padding mode is always 'reflect'  
    subtrahend = pad(subtrahend, footprint=sn_footprint, mode=sn_mode, cval=sn_cval)
    
    # compute slicers for position where the search neighbourhood sn_footprint is TRUE
    slicers = [[slice(x, (x + 1) - d if 0 != (x + 1) - d else None) for x in range(d)] for d in sn_fshape]
    slicers = [sl for sl, tv in zip(itertools.product(*slicers), sn_footprint.flat) if tv]
    
    # compute difference images and sign images for search neighbourhood elements
    ssds = [ssd(minuend, subtrahend[slicer], normalized=True, signed=signed, size=pn_size, footprint=pn_footprint, mode=pn_mode, cval=pn_cval) for slicer in slicers]
    distance = [x[0] for x in ssds]
    distance_sign = [x[1] for x in ssds]

    # compute local variance, which constitutes an approximation of local noise, out of patch-distances over the neighbourhood structure
    variance = numpy.average(distance, 0)
    variance = gaussian_filter(variance, sigma=3) #!TODO: Figure out if a fixed sigma is desirable here... I think that yes
    if 'global' == noise:
        variance = variance.sum() / float(numpy.product(variance.shape))
    # variance[variance < variance_global / 10.] = variance_global / 10. #!TODO: Should I keep this i.e. regularizing the variance to be at least 10% of the global one?
    
    # compute sls
    sls = [dist_sign * numpy.exp(-1 * (dist / variance)) for dist_sign, dist in zip(distance_sign, distance)]
    
    # convert into sls image, swapping dimensions to have varying patches in the last dimension
    return numpy.rollaxis(numpy.asarray(sls), 0, minuend.ndim + 1)

def ssd(minuend, subtrahend, normalized=True, signed=False, size=None, footprint=None, mode="reflect", cval=0.0, origin=0):
    r"""
    Computes the sum of squared difference (SSD) between patches of minuend and subtrahend.
    
    Parameters
    ----------
    minuend : array_like
        Input array from which to subtract the subtrahend.
    subtrahend : array_like
        Input array to subtract from the minuend.    
    normalized : bool, optional
        Whether the SSD of each patch should be divided through the filter size for
        normalization. Default is 'True'.
    signed : bool, optional
        Whether the accumulative sign of each patch should be returned as well. If
        'True', the second return value is a numpy.sign array, otherwise the scalar '1'.
        Default is 'False'.
    size : scalar or tuple, optional
        See footprint, below
    footprint : array, optional
        The patch over which to compute the SSD.
        Either `size` or `footprint` must be defined. `size` gives
        the shape that is taken from the input array, at every element
        position, to define the input to the filter function.
        `footprint` is a boolean array that specifies (implicitly) a
        shape, but also which of the elements within this shape will get
        passed to the filter function. Thus ``size=(n,m)`` is equivalent
        to ``footprint=np.ones((n,m))``. We adjust `size` to the number
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and `size` is 2, then the actual size used is
        (2,2,2).
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0
        
    Returns
    -------
    ssd : ndarray
        The patchwise sum of squared differences between minuend and subtrahend.
    """
    convolution_filter = average_filter if normalized else sum_filter
    output = numpy.float if normalized else minuend.dtype
    
    if signed:
        difference = minuend - subtrahend
        difference_squared = numpy.square(difference)
        distance_sign = numpy.sign(convolution_filter(numpy.sign(difference) * difference_squared, size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, output=output))
        distance = convolution_filter(difference_squared, size=size, footprint=footprint, mode=mode, cval=cval, output=output)
    else:
        distance = convolution_filter(numpy.square(minuend - subtrahend), size=size, footprint=footprint, mode=mode, cval=cval, origin=origin, output=output)
        distance_sign = 1
    
    return distance, distance_sign

def average_filter(input, size=None, footprint=None, output=None, mode="reflect", cval=0.0, origin=0):
    r"""
    Calculates a multi-dimensional average filter.

    Parameters
    ----------
    input : array-like
        input array to filter
    size : scalar or tuple, optional
        See footprint, below
    footprint : array, optional
        Either `size` or `footprint` must be defined. `size` gives
        the shape that is taken from the input array, at every element
        position, to define the input to the filter function.
        `footprint` is a boolean array that specifies (implicitly) a
        shape, but also which of the elements within this shape will get
        passed to the filter function. Thus ``size=(n,m)`` is equivalent
        to ``footprint=np.ones((n,m))``. We adjust `size` to the number
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and `size` is 2, then the actual size used is
        (2,2,2).
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    origin : scalar, optional
        The ``origin`` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    average_filter : ndarray
        Returned array of same shape as `input`.

    Notes
    -----
    Convenience implementation employing convolve.

    See Also
    --------
    scipy.ndimage.filters.convolve : Convolve an image with a kernel.
    """
    footprint = __make_footprint(input, size, footprint)
    filter_size = footprint.sum()
    
    output = _get_output(output, input)
    sum_filter(input, footprint=footprint, output=output, mode=mode, cval=cval, origin=origin)
    output /= filter_size
    return output


def sum_filter(input, size=None, footprint=None, output=None, mode="reflect", cval=0.0, origin=0):
    r"""
    Calculates a multi-dimensional sum filter.

    Parameters
    ----------
    input : array-like
        input array to filter
    size : scalar or tuple, optional
        See footprint, below
    footprint : array, optional
        Either `size` or `footprint` must be defined. `size` gives
        the shape that is taken from the input array, at every element
        position, to define the input to the filter function.
        `footprint` is a boolean array that specifies (implicitly) a
        shape, but also which of the elements within this shape will get
        passed to the filter function. Thus ``size=(n,m)`` is equivalent
        to ``footprint=np.ones((n,m))``. We adjust `size` to the number
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and `size` is 2, then the actual size used is
        (2,2,2).
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    origin : scalar, optional
        The ``origin`` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    sum_filter : ndarray
        Returned array of same shape as `input`.

    Notes
    -----
    Convenience implementation employing convolve.

    See Also
    --------
    scipy.ndimage.filters.convolve : Convolve an image with a kernel.
    """
    footprint = __make_footprint(input, size, footprint)
    slicer = [slice(None, None, -1)] * footprint.ndim 
    return convolve(input, footprint[slicer], output, mode, cval, origin)

def otsu (img, bins=64):
    r"""
    Otsu's method to find the optimal threshold separating an image into fore- and background.
    
    This rather expensive method iterates over a number of thresholds to separate the
    images histogram into two parts with a minimal intra-class variance.
    
    An increase in the number of bins increases the algorithms specificity at the cost of
    slowing it down.
    
    Parameters
    ----------
    img : array_like
        The image for which to determine the threshold.
    bins : integer
        The number of histogram bins.
        
    Returns
    -------
    otsu : float
        The otsu threshold to separate the input image into fore- and background.
    """
    # cast bins parameter to int
    bins = int(bins)
    
    # cast img parameter to scipy arrax
    img = numpy.asarray(img)
    
    # check supplied parameters
    if bins <= 1:
        raise AttributeError('At least a number two bins have to be provided.')
    
    # determine initial threshold and threshold step-length
    steplength = (img.max() - img.min()) / float(bins)
    initial_threshold = img.min() + steplength
    
    # initialize best value variables
    best_bcv = 0
    best_threshold = initial_threshold
    
    # iterate over the thresholds and find highest between class variance
    for threshold in numpy.arange(initial_threshold, img.max(), steplength):
        mask_fg = (img >= threshold)
        mask_bg = (img < threshold)
        
        wfg = numpy.count_nonzero(mask_fg)
        wbg = numpy.count_nonzero(mask_bg)
        
        if 0 == wfg or 0 == wbg: continue
        
        mfg = img[mask_fg].mean()
        mbg = img[mask_bg].mean()
        
        bcv = wfg * wbg * math.pow(mbg - mfg, 2)
        
        if bcv > best_bcv:
            best_bcv = bcv
            best_threshold = threshold
        
    return best_threshold

def local_minima(img, min_distance = 4):
    r"""
    Returns all local minima from an image.
    
    Parameters
    ----------
    img : array_like
        The image.
    min_distance : integer
        The minimal distance between the minimas in voxels. If it is less, only the lower minima is returned.
    
    Returns
    -------
    indices : sequence
        List of all minima indices.
    values : sequence
        List of all minima values.
    """
    # @TODO: Write a unittest for this.
    fits = numpy.asarray(img)
    minfits = minimum_filter(fits, size=min_distance) # default mode is reflect
    minima_mask = fits == minfits
    good_indices = numpy.transpose(minima_mask.nonzero())
    good_fits = fits[minima_mask]
    order = good_fits.argsort()
    return good_indices[order], good_fits[order]

def resample(img, hdr, target_spacing, bspline_order=3, mode='constant'):
        """
        Re-sample an image to a new voxel-spacing.
        
        Parameters
        ----------
        img : array_like
            The image.
        hdr : object
            The image header.
        target_spacing : number or sequence of numbers
            The target voxel spacing to achieve. If a single number, isotropic spacing is assumed.
        bspline_order : int
            The bspline order used for interpolation.
        mode : str
            Points outside the boundaries of the input are filled according to the given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default is 'constant'.
            
        Warning
        -------
        Voxel-spacing of input header will be modified in-place!
            
        Returns
        -------
        img : ndarray
            The re-sampled image.
        hdr : object
            The image header with the new voxel spacing.
        """
        if isinstance(target_spacing, numbers.Number):
            target_spacing = [target_spacing] * img.ndim
        
        # compute zoom values
        zoom_factors = [old / float(new) for new, old in zip(target_spacing, header.get_pixel_spacing(hdr))]
    
        # zoom image
        img = zoom(img, zoom_factors, order=bspline_order, mode=mode)
        
        # set new voxel spacing
        header.set_pixel_spacing(hdr, target_spacing)
        
        return img, hdr

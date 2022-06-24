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
# version r0.1.0
# since 2014-03-20
# status Release

# build-in modules

# third-party modules
import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage.filters import convolve1d

# own modules

# code
def immerkaer_local(input, size, output=None, mode="reflect", cval=0.0):
    r"""
    Estimate the local noise.
    
    The input image is assumed to have additive zero mean Gaussian noise. The Immerkaer
    noise estimation is applied to the image locally over a N-dimensional cube of
    side-length size. The size of the region should be sufficiently high for a stable
    noise estimation.
    
    Parameters
    ----------
    input : array_like
        Array of which to estimate the noise.
    size : integer
        The local region's side length.
    output : ndarray, optional
        The `output` parameter passes an array in which to store the
        filter output.        
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0        
        
    Returns
    -------
    sigmas : array_like
        Map of the estimated standard deviation of the images Gaussian noise per voxel.
    
    Notes
    -----
    Does not take the voxel spacing into account.
    Works good with medium to strong noise. Tends to underestimate for low noise levels.
    
    See also
    --------
    immerkaer
    """
    output = _ni_support._get_output(output, input)
    footprint = numpy.asarray([1] * size)
    
    # build nd-kernel to acquire square root of sum of squared elements
    kernel = [1, -2, 1]
    for _ in range(input.ndim - 1):
        kernel = numpy.tensordot(kernel, [1, -2, 1], 0)
    divider = numpy.square(numpy.abs(kernel)).sum() # 36 for 1d, 216 for 3D, etc.
    
    # compute laplace of input
    laplace = separable_convolution(input, [1, -2, 1], numpy.double, mode, cval)
    
    # compute factor
    factor = numpy.sqrt(numpy.pi / 2.) * 1. / ( numpy.sqrt(divider) * numpy.power(footprint.size, laplace.ndim) )
    
    # locally sum laplacian values
    separable_convolution(numpy.abs(laplace), footprint, output, mode, cval)
    
    output *= factor
    
    return output

def immerkaer(input, mode="reflect", cval=0.0):
    r"""
    Estimate the global noise.
    
    The input image is assumed to have additive zero mean Gaussian noise. Using a
    convolution with a Laplacian operator and a subsequent averaging the standard
    deviation sigma of this noise is estimated. This estimation is global i.e. the
    noise is assumed to be globally homogeneous over the image.
    
    Implementation based on [1]_.
    
        
    Immerkaer suggested a Laplacian-based 2D kernel::
    
        [[ 1, -2,  1],
         [-2,  4, -1],
         [ 1, -2, 1]]

    , which is separable and can therefore be applied by consecutive convolutions with
    the one dimensional kernel [1, -2, 1].
    
    We generalize from this 1D-kernel to an ND-kernel by applying N consecutive
    convolutions with the 1D-kernel along all N dimensions.
    
    This is equivalent with convolving the image with an ND-kernel constructed by calling
    
    >>> kernel1d = numpy.asarray([1, -2, 1])
    >>> kernel = kernel1d.copy()
    >>> for _ in range(input.ndim):
    >>>     kernel = numpy.tensordot(kernel, kernel1d, 0)
    
    Parameters
    ----------
    input : array_like
        Array of which to estimate the noise.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0        
        
    Returns
    -------
    sigma : float
        The estimated standard deviation of the images Gaussian noise.
        
    Notes
    -----
    Does not take the voxel spacing into account.
    Works good with medium to strong noise. Tends to underestimate for low noise levels.
        
    See also
    --------
    immerkaer_local
    
    References
    ----------
    .. [1] John Immerkaer, "Fast Noise Variance Estimation", Computer Vision and Image
           Understanding, Volume 64, Issue 2, September 1996, Pages 300-302, ISSN 1077-3142
    """
    # build nd-kernel to acquire square root of sum of squared elements
    kernel = [1, -2, 1]
    for _ in range(input.ndim - 1):
        kernel = numpy.tensordot(kernel, [1, -2, 1], 0)
    divider = numpy.square(numpy.abs(kernel)).sum() # 36 for 1d, 216 for 3D, etc.
    
    # compute laplace of input and derive noise sigma
    laplace = separable_convolution(input, [1, -2, 1], None, mode, cval)
    factor = numpy.sqrt(numpy.pi / 2.) * 1. / ( numpy.sqrt(divider) * numpy.prod(laplace.shape) )
    sigma = factor * numpy.abs(laplace).sum()
    
    return sigma
    
def separable_convolution(input, weights, output=None, mode="reflect", cval=0.0, origin=0):
    r"""
    Calculate a n-dimensional convolution of a separable kernel to a n-dimensional input.
    
    Achieved by calling convolution1d along the first axis, obtaining an intermediate
    image, on which the next convolution1d along the second axis is called and so on.
    
    Parameters
    ----------
    input : array_like
        Array of which to estimate the noise.
    weights : ndarray
        One-dimensional sequence of numbers.          
    output : array, optional
        The `output` parameter passes an array in which to store the
        filter output.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0.0.
        
    Returns
    -------
    output : ndarray
        Input image convolved with the supplied kernel.
    """
    input = numpy.asarray(input)
    output = _ni_support._get_output(output, input)
    axes = list(range(input.ndim))
    if len(axes) > 0:
        convolve1d(input, weights, axes[0], output, mode, cval, origin)
        for ii in range(1, len(axes)):
            convolve1d(output, weights, axes[ii], output, mode, cval, origin)
    else:
        output[...] = input[...]
    return output
    
    
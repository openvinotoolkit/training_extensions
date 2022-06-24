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
# author Alexander Ruesch
# version r0.1.1
# since 2013-08-24
# status Release

# build-in modules

# third-party modules
import numpy
from scipy.ndimage.filters import uniform_filter, sobel, maximum_filter, minimum_filter, gaussian_filter
from scipy import stats
from math import factorial

# own modules

# constants

def coarseness(image, voxelspacing = None, mask = slice(None)):
    r"""
    Takes a simple or multi-spectral image and returns the coarseness of the texture.
    
    Step1  At each pixel, compute six averages for the windows of size 2**k x 2**k,
            k=0,1,...,5, around the pixel. 
    Step2  At each pixel, compute absolute differences E between the pairs of non 
            overlapping averages in every directions.
    step3  At each pixel, find the value of k that maximises the difference Ek in either 
            direction and set the best size Sbest=2**k
    step4  Compute the coarseness feature Fcrs by averaging Sbest over the entire image.

    Parameters
    ----------
    image : array_like or list/tuple of array_like 
        A single image or a list/tuple of images (for multi-spectral case).
    voxelspacing : sequence of floats
        The side-length of each voxel.
    mask : array_like
        A binary mask for the image or a slice object
        
    Returns
    -------
    coarseness : float
        The size of coarseness of the given texture. It is basically the size of
        repeating elements in the image. 
        
    See Also
    --------
    
    
    """
    # Step1:  At each pixel (x,y), compute six averages for the windows
    # of size 2**k x 2**k, k=0,1,...,5, around the pixel.

    image = numpy.asarray(image, dtype=numpy.float32)
   
  
    # set default mask or apply given mask
    if not type(mask) is slice:
        if not type(mask[0] is slice):
            mask = numpy.array(mask, copy=False, dtype = numpy.bool)
    image = image[mask]
    
    # set default voxel spacing if not suppliec
    if None == voxelspacing:
        voxelspacing = tuple([1.] * image.ndim)
    
    if len(voxelspacing) != image.ndim:
        print("Voxel spacing and image dimensions do not fit.")
        return None
    # set padding for image border control
    padSize = numpy.asarray([(numpy.rint((2**5.0) * voxelspacing[jj]),0) for jj in range(image.ndim)]).astype(numpy.int)
    Apad = numpy.pad(image,pad_width=padSize, mode='reflect')

    # Allocate memory
    E = numpy.empty((6,image.ndim)+image.shape)

    # prepare some slicer 
    rawSlicer           = [slice(None)] * image.ndim
    slicerForImageInPad = [slice(padSize[d][0],None)for d in range(image.ndim)]

    for k in range(6):

        size_vs = tuple(numpy.rint((2**k) * voxelspacing[jj]) for jj in range(image.ndim))
        A = uniform_filter(Apad, size = size_vs, mode = 'mirror')

        # Step2: At each pixel, compute absolute differences E(x,y) between 
        # the pairs of non overlapping averages in the horizontal and vertical directions.
        for d in range(image.ndim):
            borders = numpy.rint((2**k) * voxelspacing[d])
            
            slicerPad_k_d   = slicerForImageInPad[:]
            slicerPad_k_d[d]= slice((padSize[d][0]-borders if borders < padSize[d][0] else 0),None)
            A_k_d           = A[slicerPad_k_d]

            AslicerL        = rawSlicer[:]
            AslicerL[d]     = slice(0, -borders)
            
            AslicerR        = rawSlicer[:]
            AslicerR[d]     = slice(borders, None)

            E[k,d,...] = numpy.abs(A_k_d[AslicerL] - A_k_d[AslicerR])

    # step3: At each pixel, find the value of k that maximises the difference Ek(x,y)
    # in either direction and set the best size Sbest(x,y)=2**k
    
    k_max = E.max(1).argmax(0)
    dim = E.argmax(1)
    dim_vox_space = numpy.asarray([voxelspacing[dim[k_max.flat[i]].flat[i]] for i in range(k_max.size)]).reshape(k_max.shape) 
    S = (2**k_max) * dim_vox_space

    # step4: Compute the coarseness feature Fcrs by averaging Sbest(x,y) over the entire image.
    return S.mean()

def contrast(image, mask = slice(None)):
    r"""
    Takes a simple or multi-spectral image and returns the contrast of the texture.
    
    Fcon = standard_deviation(gray_value) / (kurtosis(gray_value)**0.25)
    
    Parameters
    ----------
    image : array_like or list/tuple of array_like 
        A single image or a list/tuple of images (for multi-spectral case).
    mask : array_like
        A binary mask for the image or a slice object
    Returns
    -------
    contrast : float
        High differences in gray value distribution is represented in a high contrast value. 
        
    See Also
    --------
    
    
    """
    image = numpy.asarray(image)
    
    # set default mask or apply given mask
    if not type(mask) is slice:
        if not type(mask[0] is slice):
            mask = numpy.array(mask, copy=False, dtype = numpy.bool)
    image = image[mask]
    
    standard_deviation = numpy.std(image)
    kurtosis = stats.kurtosis(image, axis=None, bias=True, fisher=False)
    n = 0.25 # The value n=0.25 is recommended as the best for discriminating the textures.  
    
    Fcon = standard_deviation / (kurtosis**n) 
    
    return Fcon

def directionality(image, min_distance = 4, threshold = 0.1, voxelspacing = None, mask = slice(None)):
    r"""
    Takes a simple or multi-spectral image and returns the directionality of the image texture.
    It is just a value representing the strength of directionality, not the specific direction. 
    
    An edge detection is applied on the image. Then the edge strength and directional angle between
    the image axis are computed. A histogram of the directional angles is than used to calculate a
    qualitative value for directionality in ONE image layer. Note that there are n choose 2 layers
    in a n dimensional image. 
    
    Warning
    -------
    Experimental. There are still issues with finding the right maxs and mins in histogram and
    predefining the number of bins for the histogram.

    Parameters
    ----------
    image : array_like or list/tuple of array_like 
        A single image or a list/tuple of images (for multi-spectral case).
    voxelspacing : sequence of floats
        The side-length of each voxel.
    mask : array_like
        A binary mask for the image or a slice object
    min_distance : int
        minimal Distance between 2 local minima or maxima in the histogram. Default is 4.
    threshold : float
        Defines a threshold between 0 and 1. It is used to ignore angles of low edge strength
        in the histogram. Default is 0.1.

    Returns
    -------
    directionality : array
        Fdir is a value between 0 and 1. 1 represents a high directionality.
        Returns the directionality of an image in relation to one special image layer.
        The returned values are sorted like this. The axis are named v,w,x,y,z
        for a five dimensional image:
                                    w   x   y   z   v     x   y   z   v   w
        arctan(delta)| delta =    ---,---,---,---,---,  ---,---,---,---,---
                                    v   w   x   y   z     v   w   x   y   z
        There are always n choose k axis relations; n=image.ndim, k=2 (2 axis in every image layer).
        
    
    See Also
    --------
    
    """
    image = numpy.asarray(image)
    ndim = image.ndim
    # set default mask or apply given mask
    if not type(mask) is slice:
        if not type(mask[0] is slice):
            mask = numpy.array(mask, copy=False, dtype = numpy.bool)
    image = image[mask]
           
    # set default voxel spacing if not suppliec
    if None == voxelspacing:
        voxelspacing = tuple([1.] * ndim)
        
    if len(voxelspacing) != ndim:
        print("Voxel spacing and image dimensions do not fit.")
        return None
   
   # Calculate amount of combinations: n choose k, normalizing factor r and voxel spacing.    
    n = (factorial(ndim)/(2*factorial(ndim-2)))
    pi1_2 = numpy.pi/2.0
    r=1.0 / (pi1_2**2)
    vs = [slice(None,None,numpy.rint(ii)) for ii in voxelspacing]
   
    # Allocate memory, define constants
    Fdir = numpy.empty(n)

    # calculate differences by using Sobel-filter. (Maybe other filter kernel like Prewitt will do a better job)
    E = [sobel(image, axis=ndim-1-i) for i in range(ndim)]
    
    # The edge strength e(x,y) is used for thresholding.
    e = sum(E) / float(ndim)
    border = [numpy.percentile(e, 1),numpy.percentile(e, 99)]
    e[e < border[0]] = 0
    e[e > border[1]] = border[1]
    e -= border[0]
    e /= border[1]
    em = e > threshold
        
    for i in range(n):
        A = numpy.arctan((E[(i + (ndim+i)/ndim) % ndim][vs]) / (E[i%ndim][vs]+numpy.spacing(1))) # [0 , pi/2]
        A = A[em[vs]]
        # Calculate number of bins for the histogram. Watch out, this is just a work around! 
        # @TODO: Write a more stable code to prevent for minimum and maximum repetition when the same value in the Histogram appears multiple times in a row. Example: image = numpy.zeros([10,10]), image[:,::3] = 1
        bins = numpy.unique(A).size + min_distance        
        H = numpy.histogram(A, bins = bins, density=True)[0] # [0 , 1]
        H[H < numpy.percentile(H,1)] = 0.0
        H_peaks, H_valleys, H_range = find_valley_range(H)
        summe = 0.0
        for idx_ap in range(len(H_peaks)):
            for range_idx in range( H_valleys[idx_ap], H_valleys[idx_ap]+H_range[idx_ap]):
                a=range_idx % len(H)
                summe += (((pi1_2*a)/bins - (pi1_2 * H_peaks[idx_ap])/bins) **2) * H[a]
        Fdir[i] = 1.0 - r * summe 
        
    return Fdir


def local_maxima(vector,min_distance = 4, brd_mode = "wrap"):
    """
    Internal finder for local maxima .
    Returns UNSORTED indices of maxima in input vector.
    """
    fits = gaussian_filter(numpy.asarray(vector,dtype=numpy.float32),1., mode=brd_mode)
    for ii in range(len(fits)):
        if fits[ii] == fits[ii-1]:
            fits[ii-1] = 0.0
    maxfits     = maximum_filter(fits, size=min_distance, mode=brd_mode)
    maxima_mask = fits == maxfits
    maximum     = numpy.transpose(maxima_mask.nonzero())
    return numpy.asarray(maximum)

def local_minima(vector,min_distance = 4, brd_mode = "wrap"):
    """
    Internal finder for local minima .
    Returns UNSORTED indices of minima in input vector.
    """
    fits = gaussian_filter(numpy.asarray(vector,dtype=numpy.float32),1., mode=brd_mode)
    for ii in range(len(fits)):
        if fits[ii] == fits[ii-1]:
            fits[ii-1] = numpy.pi/2.0
    minfits = minimum_filter(fits, size=min_distance, mode=brd_mode)
    minima_mask = fits == minfits
    minima = numpy.transpose(minima_mask.nonzero())
    return numpy.asarray(minima)

def find_valley_range(vector, min_distance = 4):
    """
    Internal finder peaks and valley ranges.
    Returns UNSORTED indices of maxima in input vector.
    Returns range of valleys before and after maximum
    """
    
    # http://users.monash.edu.au/~dengs/resource/papers/icme08.pdf
    # find min and max with mode = wrap
    mode = "wrap"
    minima = local_minima(vector,min_distance,mode)
    maxima = local_maxima(vector,min_distance,mode)

    if len(maxima)>len(minima):
        if vector[maxima[0]] >= vector[maxima[-1]]:
            maxima=maxima[1:]
        else:
            maxima=maxima[:-1]
        
    if len(maxima)==len(minima):
        valley_range = numpy.asarray([minima[ii+1] - minima[ii] for ii in range(len(minima)-1)] + [len(vector)-minima[-1]+minima[0]])
        if minima[0] < maxima[0]:
            minima = numpy.asarray(list(minima) + [minima[0]])
        else:
            minima = numpy.asarray(list(minima) + [minima[-1]])       
    else:
        valley_range = numpy.asarray([minima[ii+1] - minima[ii] for ii in range(len(maxima))])
    
    return maxima, minima, valley_range

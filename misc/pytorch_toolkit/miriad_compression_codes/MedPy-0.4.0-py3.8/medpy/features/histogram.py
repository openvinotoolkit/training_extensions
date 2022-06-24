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
# version r0.1.3
# since 2012-03-01
# status Release

# build-in modules
import math

# third-party modules
import scipy.stats

# own modules

# constants
# the available membership functions for fuzzy histogram calculation
__MBS = ['triangular', 'trapezoid', 'gaussian', 'sigmoid']

# code
def fuzzy_histogram(a, bins=10, range=None, normed=False, membership='triangular', smoothness=None, guarantee=False):
    r"""Compute a fuzzy histogram.
    The percentage of a value's membership in a bin is computed using the selected
    membership function. This functions stays as near as possible to the `numpy.histogram`
    behaviour.
    
    Parameters
    ----------
    a : array_like
        Input data; The histogram is computed over the flattened array (with ravel()).
    bins : int
        The number of equal-width bins in the given range (10, by default).
    range : (float, float)
        The lower and upper range of the bins; If not provided, range is simply
        (a.min(), a.max()); Values outside the range are ignored.
    normed : bool
        If False, the result will contain the number of samples in each bin; If
        True, the result is the value of the probability density function at
        the bin, normalized such that the integral over the range is 1.
    membership : string
        Select the type of the fuzzy membership function; See package
        description for available options.
    smoothness : float
        The smoothness of the fuzzy function; See package
        description and the membership functions for more details.
    guarantee : bool
        Guarantee that all values contribute equally to the histogram; when this value is
        set, the range term is ignored; see package descriptions for details.
    
    Returns
    -------
    hist : array
        The values of the histogram. See normed and weights for a description of the possible semantics.
    bin_edges : array of dtype float
        Return the bin edges (length(hist)+1).
    
    Notes
    -----
    See package description for more details on the usage.
    
    Examples
    --------
    >>> import numpy as np
    >>> from medpy.features import fuzzy_histogram
    >>> a = np.asarray([1,2,3,3.2,3.4,3.5,7.5,7.6,7.8,8,9,10])
    >>> np.histogram(a, bins=4)
    (array([4, 2, 2, 4]), array([  1.  ,   3.25,   5.5 ,   7.75,  10.  ]))
    >>> fuzzy_histogram(a, bins=4)
    (array([ 3.4       ,  2.04444444,  2.04444444,  3.4       ]), array([  1.  ,   3.25,   5.5 ,   7.75,  10.  ]))
    >>> fuzzy_histogram(a, bins=4, membership='sigmoid')
    (array([ 3.34304743,  2.15613626,  2.15613626,  3.34304743]), array([  1.  ,   3.25,   5.5 ,   7.75,  10.  ]))
    
    """
    # check and prepare parameters
    a = scipy.asarray(a).ravel()
    if None == range: range = (a.min(), a.max())
    if range[1] <= range[0]: raise AttributeError('max must be larger than min in range parameter.')
    if not int == type(bins): raise AttributeError('bins must an integer.')
    if bins <= 0: raise AttributeError('bins must greater than zero.')
    if membership not in __MBS: raise AttributeError('Unknown type: {}. Must be one of {}.'.format(membership, __MBS))
    if not None == smoothness and smoothness <= 0.0: raise AttributeError('smoothness must be greater than zero.')
    
    # set default smoothness values
    if None == smoothness:
        smoothness = 0.25 if 'trapezoid' == membership else 0.5
    
    if not guarantee: # compute bin distribution in no guarantee case
        binw = (range[1] - range[0]) / float(bins)
        bins = scipy.asarray([i * binw + range[0] for i in scipy.arange(bins + 1)])
    else: # compute bin distribution for guarantee case
        bins_core = bins - 2 * int(math.ceil(smoothness))
        if bins_core <= 0: raise AttributeError('bins to few to guarantee removing boundary effect.')
        binw = (range[1] - range[0]) / float(bins_core)
        range = (range[0] - int(math.ceil(smoothness)) * binw, range[1] + int(math.ceil(smoothness)) * binw)
        bins = scipy.asarray([i * binw + range[0] for i in scipy.arange(bins + 1)])
    
    # create membership function (centered at 0)
    if 'triangular' == membership:
        membership = triangular_membership(0, binw, smoothness)
    elif 'trapezoid' == membership:
        membership = trapezoid_membership(0, binw, smoothness)
    elif 'gaussian' == membership:
        membership = gaussian_membership(0, binw, smoothness)
    elif 'sigmoid' == membership:
        membership = sigmoidal_difference_membership(0, binw, smoothness)

    # compute histogram i.e. memberships of values across neighbourhood (determined by smoothness)
    neighbourhood = int(math.ceil(smoothness))
    l = len(bins) - 2
    histogram = scipy.zeros(l + 1)
    m = range[0]
    for v in a: # for each value
        idx = min(l, int((v - m) / binw))
        for i in scipy.arange(max(0, idx - neighbourhood), min(l + 1, idx + neighbourhood + 1)): # for crips bin neighbourhood
            start = bins[i]
            histogram[i] += membership(v - start - 0.5 * binw) # adjust v for evaluation on zero-centered membership function

    # normalize
    if normed: histogram /= float(sum(histogram))
    
    return histogram, bins
    
# //////////////////// #
# Membership functions #
# //////////////////// #
# see http://www.atp.ruhr-uni-bochum.de/rt1/syscontrol/node117.html for graphs    

def triangular_membership(bin_center, bin_width, smoothness = 0.5):
    r"""
    Create a triangular membership function for a fuzzy histogram bin.
    
    Parameters
    ----------
    bin_center : number
        The center of the bin of which to compute the membership function.
    bin_width : number
        The width of a single bin (all expected to be of equal width).
    smoothness : number, optional
        The smoothness of the function; determines the neighbourhood affected.
        See below and `fuzzy_histogram` for a more detailed explanation
    
    Returns
    -------
    triangular_membership : function
        A triangular membership function centered on the bin.
    
    Notes
    -----
    For the triangular function the smoothness factor has to be 0.5. Lower values
    are accepted, but then the function assumes the shape of the trapezium membership
    function. Higher values lead to an exception.
    
    The triangular membership function is defined as

    .. math::
        
        \mu_{\triangle}(x) =
          \left\{
            \begin{array}{ll}
              0, & x<a, x>c\\
              \frac{x-a}{b-a}, & a\leq x\leq b \\
              \frac{c-x}{c-b}, & b<x\leq c\\
            \end{array}
          \right.

    where :math:`a` is the left border, :math:`c` the right border and :math:`b` the center of the triangular
    function. The height of the triangle is chosen such, that all values contribute with
    exactly one.
    
    The standard triangular function (:math:`smoothness = 0.5`) is displayed in the following
    figure
    
    .. .. image:: images/triangular_01.png
    
    "Triangular functions (1)"
    
    where the bin width is :math:`2` with centers at :math:`-2`, :math:`0` and :math:`2`.
    """
    if smoothness > 0.5: raise AttributeError('the triangular/trapezium membership functions supports only smoothnesses between 1/10 and 1/2.')
    if smoothness < 0.5: return trapezoid_membership(bin_center, bin_width, smoothness)
    
    a = bin_center - bin_width
    b = float(bin_center)
    c = bin_center + bin_width
    
    def fun(x):
        if x < a or x > c: return 0
        elif x <= b: return (x-a)/(b-a)
        else: return (c-x)/(c-b)
    return fun
    
def trapezoid_membership(bin_center, bin_width, smoothness):
    r"""Create a trapezium membership function for a fuzzy histogram bin.
    
    Parameters
    ----------
    bin_center : number
        The center of the bin of which to compute the membership function.
    bin_width : number
        The width of a single bin (all expected to be of equal width).
    smoothness : number, optional
        The smoothness of the function; determines the neighbourhood affected.
        See below and `fuzzy_histogram` for a more detailed explanation
        
    Returns
    -------
    trapezoid_membership : function
        A trapezoidal membership function centered on the bin.
    
    Notes
    -----
    For the trapezium function the smoothness factor can be between >0.0 and <0.5.
    Higher values are excepted, but then the function assumes the shape of the triangular
    membership function. A value of 0.0 would make the histogram behave like a crisp one.
    
    The trapezium membership function is defined as
    
    .. math::
    
        \mu_{trapez}(x) =
          \left\{
            \begin{array}{ll}
              0, & x<a, x>d\\
              \frac{x-a}{b-a}, & a\leq x\leq b \\
              1, & b<x<c\\
              \frac{d-x}{d-c}, & c\leq x\leq d\\
            \end{array}
          \right.
    
    where :math:`a` is the left lower border, :math:`b` the left upper border, :math:`c` the right upper border
    and :math:`d` the right lower border of the trapezium.
    
    A smoothness term of 0.1 makes the trapezium function reach by :math:`0.1 * bin\_width` into
    the areas of the adjunct bins, as can be observed in the following figure
    
    .. .. image:: images/trapezium_02.png
    
    "Trapezium functions (1)"
    
    where the bin width is 2 with centers at -2, 0 and 2.
    
    Increasing the smoothness term toward 0.5, the function starts to resemble the
    triangular membership function, which in fact it becomes for any :math:`smoothness >= 0.5`.
    The behavior can be observed in the following graph with :math:`smoothness=0.4` 
    
    .. .. image:: images/trapezium_01.png
    
    "Trapezium functions (2)"
    
    Lowering the smoothness toward 0.0, on the other hand, leads the trapezium function
    to behave more and more like a crisp histogram membership, which in fact it becomes
    at a smoothness of 0.0. The following figure, where the smoothness term is near zero,
    illustrates this behaviour
    
    .. .. image:: images/trapezium_03.png
    
    "Trapezium functions (3)"
    
    """
    # special case of high smoothness
    if smoothness < 1./10: raise AttributeError('the triangular/trapezium membership functions supports only smoothnesses between 1/10 and 1/2.')
    if smoothness >= 0.5: return triangular_membership(bin_center, bin_width, smoothness)

    a = bin_center - (smoothness + 0.5) * bin_width 
    b = bin_center - (0.5 - smoothness) * bin_width
    c = bin_center + (0.5 - smoothness) * bin_width
    d = bin_center + (smoothness + 0.5) * bin_width
     
    def fun(x):
        if x < a or x > d: return 0
        elif x <= b: return (x-a)/float(b-a)
        elif x <= c: return 1
        else: return (d-x)/float(d-c)
    return fun

def gaussian_membership(bin_center, bin_width, smoothness):
    r"""Create a gaussian membership function for a fuzzy histogram bin.
    
    Parameters
    ----------
    bin_center : number
        The center of the bin of which to compute the membership function.
    bin_width : number
        The width of a single bin (all expected to be of equal width).
    smoothness : number, optional
        The smoothness of the function; determines the neighbourhood affected.
        See below and `fuzzy_histogram` for a more detailed explanation
        
    Returns
    -------
    gaussian_membership : function
        The cumulative density function of the desired gaussian.

    Notes
    -----
    Since the gaussian membership function is infinite, it is
    not actually true that it does not contribute to bins outside of the neighbourhood
    range. But the contribution is so marginal (:math:`eps <= 0.001` per value) that it can be
    safely ignored.
    
    The gaussian membership function is defined as
    
    .. math::
    
        \mu_{gauss}(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\zeta)^2}{2\sigma^2}}

    Since the gaussian distributions can not be formed to sum up to one at each point of
    the x-axis, their cumulative density functions (CDF) are used instead. For more
    details on CDF see http://en.wikipedia.org/wiki/Normal_distribution .
    
    The gaussian and therefore the CDF are centered above the requested value instead of
    the bin center. Then the CDF value for the left side of the bin is subtracted from
    the CDF value returned for the right side. The result is the integral under the
    gaussian with :math:`\mu/\zeta = value` with the bin-sides as the integral borders.
    
    This approach might seem a little bit unintuitive, but is the best possible for
    gaussian membership functions. The following graph gives a graphical example of the
    computation of each values bin membership
    
    .. .. image:: images/gaussian_01.png 
    
    "Trapezium functions (1)"
    
    where the bin_width is 1, one bin between each of the x tics (e.g. [-1, 0], [0, 1],
    etc.). The value which membership should be computed is marked by a yellow bar at
    :math:`x = 0.3`. Its membership in each bin is defined by the integral under the gaussian
    centered at the value (i.e. 0.3). The purple area therefore defines its membership in
    the [-2,-1] bin, the red area its membership in the [-1,0] bin, etc.
    Since the gaussian is guaranteed to have an infinite integral of 1, the some of the
    contributions of a value to all bins is one.

    For computation the function normalizes all values to a bin_width of 1, which can
    introduce marginal rounding errors.
    """
    if smoothness > 10 or smoothness < 1./10: raise AttributeError('the gaussian membership function supports only smoothnesses between 1/10 and 5.')
    
    bin_width = float(bin_width)
    bin_center = bin_center / bin_width
    start = bin_center - 0.5
    end = bin_center + 0.5
    sigma = _gaussian_membership_sigma(smoothness)

    def fun(x):
        return scipy.stats.norm.cdf(end, x / bin_width, sigma) - scipy.stats.norm.cdf(start, x / bin_width, sigma) # x, mu, sigma
    
    return fun

def _gaussian_membership_sigma(smoothness, eps = 0.0005): # 275us @ smothness=10
    r"""Compute the sigma required for a gaussian, such that in a neighbourhood of
    smoothness the maximum error is 'eps'.
    The error is here the difference between the clipped integral and one.
    """
    error = 0
    deltas = [0.1, 0.01, 0.001, 0.0001]
    sigma = smoothness * 0.3
    point = -1. * (smoothness + 0.5)
    for delta in deltas:
        while error < eps:
            sigma += delta
            error = scipy.stats.norm.cdf(0.5, point, sigma) - scipy.stats.norm.cdf(-0.5, point, sigma) # x, mu, sigma
        sigma -= delta
    return sigma

def sigmoidal_difference_membership(bin_center, bin_width, smoothness):
    r"""Create the difference of two sigmoids as membership function for a fuzzy histogram bin.
    
    Parameters
    ----------
    bin_center : number
        The center of the bin of which to compute the membership function.
    bin_width : number
        The width of a single bin (all expected to be of equal width).
    smoothness : number, optional
        The smoothness of the function; determines the neighbourhood affected.
        See below and `fuzzy_histogram` for a more detailed explanation
        
    Returns
    -------
    sigmoidal_difference_membership : function
        A sigmoidal difference membership function centered on the bin.
    
    Notes
    -----
    Since the sigmoidal membership function is infinite, it is
    not actually true that it does not contribute to bins outside of the neighbourhood
    range. But the contribution is so marginal (eps <= 0.001 per value) that it can be
    safely ignored.
    
    The sigmoidal membership function is defined as
    
    .. math::
        
        \mu_{sigmoid}(x) = \left[1+e^{-\alpha_1 (x-\zeta_1)}\right]^{-1} - \left[1+e^{-\alpha_2 (x-\zeta_2)}\right]^{-1}

    where :math:`\alpha_1 = \alpha_2 = \alpha` is computed throught the smoothness term
    and :math:`\zeta_1` and :math:`\zeta_2` constitute the left resp. right borders of the bin.
    
    The following figure shows three sigmoidal membership functions for bins at the
    centers -2, -0 and 2 with a bin width of 2 and a smoothness of 2:
    
    .. .. image:: images/sigmoid_01.png
    
    "Sigmoidal functions (1)"
    
    The central (green) membership functions extends to its up till the second bin
    (centered around -4) and the same to the right (until the bin centered around +4).
    Therefore all values from -5 to +5 are considered for membership in this bin. Values
    out of this range would only contribute marginally to this bin.
    Furthermore it is inteligable that the sum of all membership functions at each point
    is equal to 1, therefore all values are equally represented (i.e. contribute with 1
    to the overall histogram).
    
    The influence of the smoothness term can be observed in the following figure:
    
    .. .. image:: images/sigmoid_02.png
    
    "Sigmoidal functions (2)"
    
    Here smoothness has been chosen to be 1. The green function therefore extends just
    into the directly adjunct bins to its left and right.
    
    """
    if smoothness > 10 or smoothness < 1./10: raise AttributeError('the sigmoidal membership function supports only smoothnesses between 1/10 and 10.')
    
    # compute the alpha that will give a contribution to the next bins right and left
    alpha_nbh1 = 8. / bin_width # experimental value
    # compute the alpha that results in the desired smoothness level 
    alpha = alpha_nbh1 / smoothness
    
    def fun(x):
        sigmoid1 = 1 + math.exp(-1. * alpha * (x - (bin_center - 0.5 * bin_width)))
        sigmoid2 = 1 + math.exp(-1. * alpha * (x - (bin_center + 0.5 * bin_width)))
        return math.pow(sigmoid1, -1) - math.pow(sigmoid2, -1)
    return fun
    
#def generalized_bell_membership(alpha, beta, zeta):
#    """
#    Create a generalized bell function as membership function for a fuzzy histogram bin.
#    
#    @param alpha controls the width of the plateau
#    @param beta controls the width of the base
#    @param zeta the center of the function
#    
#    Recommended values are:
#        - alpha: bin-width/2
#        - beta: bin-width/2
#        - zeta: bin center
#    
#    The bell membership function is defined as    
#    \f[
#     \mu_{bell}(x) = \left[1+\left|\frac{x-\zeta}{\alpha}\right|^{2\beta}\right]^{-1}
#    \f]
#    """
#    def fun(x):
#        try:
#            return math.pow(1 + math.pow(abs((x - zeta)/float(alpha)), 2. * beta), -1)
#        except Exception as e:
#            print x, zeta, alpha, beta
#            raise e
#    return fun

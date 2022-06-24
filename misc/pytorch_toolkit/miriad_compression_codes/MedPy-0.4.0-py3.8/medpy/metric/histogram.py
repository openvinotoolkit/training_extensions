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
# since 2011-12-01
# status Release

# build-in modules
import math

# third-party modules
import scipy

# own modules

# code
# ////////////////////////////// #
# Bin-by-bin comparison measures #
# ////////////////////////////// #

def minowski(h1, h2, p = 2): # 46..45..14,11..43..44 / 45 us for p=int(-inf..-24..-1,1..24..inf) / float @array, +20 us @list \w 100 bins
    r"""
    Minowski distance.
    
    With :math:`p=2` equal to the Euclidean distance, with :math:`p=1` equal to the Manhattan distance,
    and the Chebyshev distance implementation represents the case of :math:`p=\pm inf`.
    
    The Minowksi distance between two histograms :math:`H` and :math:`H'` of size :math:`m` is
    defined as:
    
    .. math::
    
        d_p(H, H') = \left(\sum_{m=1}^M|H_m - H'_m|^p  
            \right)^{\frac{1}{p}}

    *Attributes:*
    
    - a real metric
    
    *Attributes for normalized histograms:*
    
    - :math:`d(H, H')\in[0, \sqrt[p]{2}]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*
    
    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*
    
    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram.
    p : float
        The :math:`p` value in the Minowksi distance formula.
    
    Returns
    -------
    minowski : float
        Minowski distance.
    
    Raises
    ------
    ValueError
        If ``p`` is zero.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    if 0 == p: raise ValueError('p can not be zero')
    elif int == type(p):
        if p > 0 and p < 25: return __minowski_low_positive_integer_p(h1, h2, p)
        elif p < 0 and p > -25: return __minowski_low_negative_integer_p(h1, h2, p)
    return math.pow(scipy.sum(scipy.power(scipy.absolute(h1 - h2), p)), 1./p)

def __minowski_low_positive_integer_p(h1, h2, p = 2): # 11..43 us for p = 1..24 \w 100 bins
    """
    A faster implementation of the Minowski distance for positive integer < 25.
    @note do not use this function directly, but the general @link minowski() method.
    @note the passed histograms must be scipy arrays.
    """
    mult = scipy.absolute(h1 - h2)
    dif = mult
    for _ in range(p - 1): dif = scipy.multiply(dif, mult)
    return math.pow(scipy.sum(dif), 1./p)

def __minowski_low_negative_integer_p(h1, h2, p = 2): # 14..46 us for p = -1..-24 \w 100 bins
    """
    A faster implementation of the Minowski distance for negative integer > -25.
    @note do not use this function directly, but the general @link minowski() method.
    @note the passed histograms must be scipy arrays.
    """
    mult = scipy.absolute(h1 - h2)
    dif = mult
    for _ in range(-p + 1): dif = scipy.multiply(dif, mult)
    return math.pow(scipy.sum(1./dif), 1./p)

def manhattan(h1, h2): # # 7 us @array, 31 us @list \w 100 bins
    r"""
    Equal to Minowski distance with :math:`p=1`.
    
    See also
    --------
    minowski
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return scipy.sum(scipy.absolute(h1 - h2))

def euclidean(h1, h2): # 9 us @array, 33 us @list \w 100 bins
    r"""
    Equal to Minowski distance with :math:`p=2`.
    
    See also
    --------
    minowski
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return math.sqrt(scipy.sum(scipy.square(scipy.absolute(h1 - h2))))

def chebyshev(h1, h2): # 12 us @array, 36 us @list \w 100 bins
    r"""
    Chebyshev distance.
    
    Also Tchebychev distance, Maximum or :math:`L_{\infty}` metric; equal to Minowski
    distance with :math:`p=+\infty`. For the case of :math:`p=-\infty`, use `chebyshev_neg`.
    
    The Chebyshev distance between two histograms :math:`H` and :math:`H'` of size :math:`m` is
    defined as:
    
    .. math::
    
        d_{\infty}(H, H') = \max_{m=1}^M|H_m-H'_m|
    
    *Attributes:*
    
    - semimetric (triangle equation satisfied?)
    
    *Attributes for normalized histograms:*
    
    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*
    
    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*
    
    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram.
    
    Returns
    -------
    chebyshev : float
        Chebyshev distance.
    
    See also
    --------
    minowski, chebyshev_neg
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return max(scipy.absolute(h1 - h2))

def chebyshev_neg(h1, h2): # 12 us @array, 36 us @list \w 100 bins
    r"""
    Chebyshev negative distance.
    
    Also Tchebychev distance, Minimum or :math:`L_{-\infty}` metric; equal to Minowski
    distance with :math:`p=-\infty`. For the case of :math:`p=+\infty`, use `chebyshev`.
    
    The Chebyshev distance between two histograms :math:`H` and :math:`H'` of size :math:`m` is
    defined as:
    
    .. math::
    
        d_{-\infty}(H, H') = \min_{m=1}^M|H_m-H'_m|
    
    *Attributes:*

    - semimetric (triangle equation satisfied?)
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram.
    
    Returns
    -------
    chebyshev_neg : float
        Chebyshev negative distance.
    
    See also
    --------
    minowski, chebyshev
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return min(scipy.absolute(h1 - h2))

def histogram_intersection(h1, h2): # 6 us @array, 30 us @list \w 100 bins
    r"""
    Calculate the common part of two histograms.
    
    The histogram intersection between two histograms :math:`H` and :math:`H'` of size :math:`m` is
    defined as:
    
    .. math::
    
        d_{\cap}(H, H') = \sum_{m=1}^M\min(H_m, H'_m)
    
    *Attributes:*

    - a real metric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 1`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    histogram_intersection : float
        Intersection between the two histograms.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return scipy.sum(scipy.minimum(h1, h2))

def histogram_intersection_1(h1, h2): # 7 us @array, 31 us @list \w 100 bins
    r"""
    Turns the histogram intersection similarity into a distance measure for normalized,
    positive histograms.
    
    .. math::
    
        d_{\bar{\cos}}(H, H') = 1 - d_{\cap}(H, H')
    
    See `histogram_intersection` for the definition of :math:`d_{\cap}(H, H')`.
    
    *Attributes:*

    - semimetric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    histogram_intersection : float
        Intersection between the two histograms.
    """
    return 1. - histogram_intersection(h1, h2)

def relative_deviation(h1, h2): # 18 us @array, 42 us @list \w 100 bins
    r"""
    Calculate the deviation between two histograms.
    
    The relative deviation between two histograms :math:`H` and :math:`H'` of size :math:`m` is
    defined as:
    
    .. math::
    
        d_{rd}(H, H') =
            \frac{
                \sqrt{\sum_{m=1}^M(H_m - H'_m)^2}
              }{
                \frac{1}{2}
                \left(
                    \sqrt{\sum_{m=1}^M H_m^2} +
                    \sqrt{\sum_{m=1}^M {H'}_m^2}
                \right)
              }
    
    *Attributes:*

    - semimetric (triangle equation satisfied?)
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, \sqrt{2}]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[0, 2]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable    
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram, same bins as ``h1``.
    
    Returns
    -------
    relative_deviation : float
        Relative deviation between the two histograms.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    numerator = math.sqrt(scipy.sum(scipy.square(h1 - h2)))
    denominator = (math.sqrt(scipy.sum(scipy.square(h1))) + math.sqrt(scipy.sum(scipy.square(h2)))) / 2.
    return numerator / denominator

def relative_bin_deviation(h1, h2): # 79 us @array, 104 us @list \w 100 bins
    r"""
    Calculate the bin-wise deviation between two histograms.
    
    The relative bin deviation between two histograms :math:`H` and :math:`H'` of size
    :math:`m` is defined as:
    
    .. math::
    
        d_{rbd}(H, H') = \sum_{m=1}^M
            \frac{
                \sqrt{(H_m - H'_m)^2}
              }{
                \frac{1}{2}
                \left(
                    \sqrt{H_m^2} +
                    \sqrt{{H'}_m^2}
                \right)
              }
    
    *Attributes:*

    - a real metric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable 
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram, same bins as ``h1``.
    
    Returns
    -------
    relative_bin_deviation : float
        Relative bin deviation between the two histograms.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    numerator = scipy.sqrt(scipy.square(h1 - h2))
    denominator = (scipy.sqrt(scipy.square(h1)) + scipy.sqrt(scipy.square(h2))) / 2.
    old_err_state = scipy.seterr(invalid='ignore') # divide through zero only occurs when the bin is zero in both histograms, in which case the division is 0/0 and leads to (and should lead to) 0
    result = numerator / denominator
    scipy.seterr(**old_err_state)
    result[scipy.isnan(result)] = 0 # faster than scipy.nan_to_num, which checks for +inf and -inf also
    return scipy.sum(result)

def chi_square(h1, h2): # 23 us @array, 49 us @list \w 100
    r"""
    Chi-square distance.
    
    Measure how unlikely it is that one distribution (histogram) was drawn from the
    other. The Chi-square distance between two histograms :math:`H` and :math:`H'` of size
    :math:`m` is defined as:
    
    .. math::
    
        d_{\chi^2}(H, H') = \sum_{m=1}^M
            \frac{
                (H_m - H'_m)^2
            }{
                H_m + H'_m
            }
    
    *Attributes:*

    - semimetric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 2]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable     
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram.
    
    Returns
    -------
    chi_square : float
        Chi-square distance.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    old_err_state = scipy.seterr(invalid='ignore') # divide through zero only occurs when the bin is zero in both histograms, in which case the division is 0/0 and leads to (and should lead to) 0
    result = scipy.square(h1 - h2) / (h1 + h2)
    scipy.seterr(**old_err_state)
    result[scipy.isnan(result)] = 0 # faster than scipy.nan_to_num, which checks for +inf and -inf also
    return scipy.sum(result)

    
def kullback_leibler(h1, h2): # 83 us @array, 109 us @list \w 100 bins
    r"""
    Kullback-Leibler divergence.
    
    Compute how inefficient it would to be code one histogram into another.
    Actually computes :math:`\frac{d_{KL}(h1, h2) + d_{KL}(h2, h1)}{2}` to achieve symmetry.
    
    The Kullback-Leibler divergence between two histograms :math:`H` and :math:`H'` of size
    :math:`m` is defined as:
    
    .. math::
    
        d_{KL}(H, H') = \sum_{m=1}^M H_m\log\frac{H_m}{H'_m}
    
    *Attributes:*

    - quasimetric (but made symetric)
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
        
    Parameters
    ----------
    h1 : sequence
        The first histogram, where h1[i] > 0 for any i such that h2[i] > 0, normalized.
    h2 : sequence
        The second histogram, where h2[i] > 0 for any i such that h1[i] > 0, normalized, same bins as ``h1``.
    
    Returns
    -------
    kullback_leibler : float
        Kullback-Leibler divergence.

    """
    old_err_state = scipy.seterr(divide='raise')
    try:
        h1, h2 = __prepare_histogram(h1, h2)
        result = (__kullback_leibler(h1, h2) + __kullback_leibler(h2, h1)) / 2.
        scipy.seterr(**old_err_state)
        return result
    except FloatingPointError:
        scipy.seterr(**old_err_state)
        raise ValueError('h1 can only contain zero values where h2 also contains zero values and vice-versa')
    
def __kullback_leibler(h1, h2): # 36.3 us
    """
    The actual KL implementation. @see kullback_leibler() for details.
    Expects the histograms to be of type scipy.ndarray.
    """
    result = h1.astype(scipy.float_)
    mask = h1 != 0
    result[mask] = scipy.multiply(h1[mask], scipy.log(h1[mask] / h2[mask]))
    return scipy.sum(result)
       
def jensen_shannon(h1, h2): # 85 us @array, 110 us @list \w 100 bins
    r"""
    Jensen-Shannon divergence.
    
    A symmetric and numerically more stable empirical extension of the Kullback-Leibler
    divergence.
    
    The Jensen Shannon divergence between two histograms :math:`H` and :math:`H'` of size
    :math:`m` is defined as:
    
    .. math::
    
        d_{JSD}(H, H') =
            \frac{1}{2} d_{KL}(H, H^*) +
            \frac{1}{2} d_{KL}(H', H^*)
    
    with :math:`H^*=\frac{1}{2}(H + H')`.
    
    *Attributes:*

    - semimetric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[0, \infty)`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram, same bins as ``h1``.
    
    Returns
    -------
    jensen_shannon : float
        Jensen-Shannon divergence.    

    """
    h1, h2 = __prepare_histogram(h1, h2)
    s = (h1 + h2) / 2.
    return __kullback_leibler(h1, s) / 2. + __kullback_leibler(h2, s) / 2.
    
def fidelity_based(h1, h2): # 25 us @array, 51 us @list \w 100 bins
    r"""
    Fidelity based distance.
    
    Also Bhattacharyya distance; see also the extensions `noelle_1` to `noelle_5`.
    
    The metric between two histograms :math:`H` and :math:`H'` of size :math:`m` is defined as:
    
    .. math::
    
        d_{F}(H, H') = \sum_{m=1}^M\sqrt{H_m * H'_m}
    
    
    *Attributes:*

    - not a metric, a similarity
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 1`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    fidelity_based : float
        Fidelity based distance.
    
    Notes
    -----
    The fidelity between two histograms :math:`H` and :math:`H'` is the same as the
    cosine between their square roots :math:`\sqrt{H}` and :math:`\sqrt{H'}`.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    result = scipy.sum(scipy.sqrt(h1 * h2))
    result = 0 if 0 > result else result # for rounding errors
    result = 1 if 1 < result else result # for rounding errors
    return result

def noelle_1(h1, h2): # 26 us @array, 52 us @list \w 100 bins
    r"""
    Extension of `fidelity_based` proposed by [1]_.
    
    .. math::
    
        d_{\bar{F}}(H, H') = 1 - d_{F}(H, H')
    
    See `fidelity_based` for the definition of :math:`d_{F}(H, H')`.
    
    *Attributes:*

    - semimetric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    fidelity_based : float
        Fidelity based distance.
    
    References
    ----------
    .. [1] M. Noelle "Distribution Distance Measures Applied to 3-D Object Recognition", 2003
    """
    return 1. - fidelity_based(h1, h2)

def noelle_2(h1, h2): # 26 us @array, 52 us @list \w 100 bins
    r"""
    Extension of `fidelity_based` proposed by [1]_.
    
    .. math::
        
        d_{\sqrt{1-F}}(H, H') = \sqrt{1 - d_{F}(H, H')}
    
    See `fidelity_based` for the definition of :math:`d_{F}(H, H')`.
    
    *Attributes:*

    - metric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    fidelity_based : float
        Fidelity based distance.
    
    References
    ----------
    .. [1] M. Noelle "Distribution Distance Measures Applied to 3-D Object Recognition", 2003
    """
    return math.sqrt(1. - fidelity_based(h1, h2))

def noelle_3(h1, h2): # 26 us @array, 52 us @list \w 100 bins
    r"""
    Extension of `fidelity_based` proposed by [1]_.
    
    .. math::
    
        d_{\log(2-F)}(H, H') = \log(2 - d_{F}(H, H'))
    
    See `fidelity_based` for the definition of :math:`d_{F}(H, H')`.
        
    *Attributes:*

    - semimetric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, log(2)]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    fidelity_based : float
        Fidelity based distance.
    
    References
    ----------
    .. [1] M. Noelle "Distribution Distance Measures Applied to 3-D Object Recognition", 2003
    """
    return math.log(2 - fidelity_based(h1, h2))

def noelle_4(h1, h2): # 26 us @array, 52 us @list \w 100 bins
    r"""
    Extension of `fidelity_based` proposed by [1]_.
    
    .. math::
    
        d_{\arccos F}(H, H') = \frac{2}{\pi} \arccos d_{F}(H, H')
    
    See `fidelity_based` for the definition of :math:`d_{F}(H, H')`.
            
    *Attributes:*

    - metric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    fidelity_based : float
        Fidelity based distance.
    
    References
    ----------
    .. [1] M. Noelle "Distribution Distance Measures Applied to 3-D Object Recognition", 2003
    """
    return 2. / math.pi * math.acos(fidelity_based(h1, h2))

def noelle_5(h1, h2): # 26 us @array, 52 us @list \w 100 bins
    r"""
    Extension of `fidelity_based` proposed by [1]_.
    
    .. math::
    
        d_{\sin F}(H, H') = \sqrt{1 -d_{F}^2(H, H')}
    
    See `fidelity_based` for the definition of :math:`d_{F}(H, H')`.
                
    *Attributes:*

    - metric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
    
    Returns
    -------
    fidelity_based : float
        Fidelity based distance.
    
    References
    ----------
    .. [1] M. Noelle "Distribution Distance Measures Applied to 3-D Object Recognition", 2003
    """
    return math.sqrt(1 - math.pow(fidelity_based(h1, h2), 2))


def cosine_alt(h1, h2): # 17 us @array, 42 us @list \w 100 bins
    r"""
    Alternative implementation of the `cosine` distance measure.
    
    Notes
    -----
    Under development.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return -1 * float(scipy.sum(h1 * h2)) / (scipy.sum(scipy.power(h1, 2)) * scipy.sum(scipy.power(h2, 2)))

def cosine(h1, h2): # 17 us @array, 42 us @list \w 100 bins
    r"""
    Cosine simmilarity.
    
    Compute the angle between the two histograms in vector space irrespective of their
    length. The cosine similarity between two histograms :math:`H` and :math:`H'` of size
    :math:`m` is defined as:
    
    .. math::
    
        d_{\cos}(H, H') = \cos\alpha = \frac{H * H'}{\|H\| \|H'\|} = \frac{\sum_{m=1}^M H_m*H'_m}{\sqrt{\sum_{m=1}^M H_m^2} * \sqrt{\sum_{m=1}^M {H'}_m^2}}
    
    
    *Attributes:*

    - not a metric, a similarity
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 1`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[-1, 1]`
    - :math:`d(H, H) = 1`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable    
        
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram, same bins as ``h1``.
    
    Returns
    -------
    cosine : float
        Cosine simmilarity.
        
    Notes
    -----
    The resulting similarity ranges from -1 meaning exactly opposite, to 1 meaning
    exactly the same, with 0 usually indicating independence, and in-between values
    indicating intermediate similarity or dissimilarity.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    return scipy.sum(h1 * h2) / math.sqrt(scipy.sum(scipy.square(h1)) * scipy.sum(scipy.square(h2)))

def cosine_1(h1, h2): # 18 us @array, 43 us @list \w 100 bins
    r"""
    Cosine simmilarity.
    
    Turns the cosine similarity into a distance measure for normalized, positive
    histograms.
    
    .. math::
    
        d_{\bar{\cos}}(H, H') = 1 - d_{\cos}(H, H')
    
    See `cosine` for the definition of :math:`d_{\cos}(H, H')`.
    
    *Attributes:*

    - metric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
        
        Returns
    -------
    cosine : float
        Cosine distance.
    """
    return 1. - cosine(h1, h2)

def cosine_2(h1, h2): # 19 us @array, 44 us @list \w 100 bins
    r"""
    Cosine simmilarity.
    
    Turns the cosine similarity into a distance measure for normalized, positive
    histograms.
    
    .. math::
    
        d_{\bar{\cos}}(H, H') = 1 - \frac{2*\arccos d_{\cos}(H, H')}{pi}
    
    See `cosine` for the definition of :math:`d_{\cos}(H, H')`.
    
    *Attributes:*

    - metric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - not applicable
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram, normalized.
    h2 : sequence
        The second histogram, normalized, same bins as ``h1``.
        
        Returns
    -------
    cosine : float
        Cosine distance. 
    """
    return 1. - (2 * cosine(h1, h2)) / math.pi

def correlate(h1, h2): # 31 us @array, 55 us @list \w 100 bins
    r"""
    Correlation between two histograms.
    
    The histogram correlation between two histograms :math:`H` and :math:`H'` of size :math:`m`
    is defined as:
    
    .. math::
    
        d_{corr}(H, H') = 
        \frac{
            \sum_{m=1}^M (H_m-\bar{H}) \cdot (H'_m-\bar{H'})
        }{
            \sqrt{\sum_{m=1}^M (H_m-\bar{H})^2 \cdot \sum_{m=1}^M (H'_m-\bar{H'})^2}
        }
    
    with :math:`\bar{H}` and :math:`\bar{H'}` being the mean values of :math:`H` resp. :math:`H'`
        
    *Attributes:*

    - not a metric, a similarity
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[-1, 1]`
    - :math:`d(H, H) = 1`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[-1, 1]`
    - :math:`d(H, H) = 1`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram, same bins as ``h1``.
    
    Returns
    -------
    correlate : float
        Correlation between the histograms.
        
    Notes
    -----
    Returns 0 if one of h1 or h2 contain only zeros.
    
    """
    h1, h2 = __prepare_histogram(h1, h2)
    h1m = h1 - scipy.sum(h1) / float(h1.size)
    h2m = h2 - scipy.sum(h2) / float(h2.size)
    a = scipy.sum(scipy.multiply(h1m, h2m))
    b = math.sqrt(scipy.sum(scipy.square(h1m)) * scipy.sum(scipy.square(h2m)))
    return 0 if 0 == b else a / b

def correlate_1(h1, h2): # 32 us @array, 56 us @list \w 100 bins
    r"""
    Correlation distance.
    
    Turns the histogram correlation into a distance measure for normalized, positive
    histograms.
    
    .. math::
    
        d_{\bar{corr}}(H, H') = 1-\frac{d_{corr}(H, H')}{2}.
    
    See `correlate` for the definition of :math:`d_{corr}(H, H')`.
    
    *Attributes:*

    - semimetric
    
    *Attributes for normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-normalized histograms:*

    - :math:`d(H, H')\in[0, 1]`
    - :math:`d(H, H) = 0`
    - :math:`d(H, H') = d(H', H)`
    
    *Attributes for not-equal histograms:*

    - not applicable
    
    Parameters
    ----------
    h1 : sequence
        The first histogram.
    h2 : sequence
        The second histogram, same bins as ``h1``.
    
    Returns
    -------
    correlate : float
        Correlation distnace between the histograms.
        
    Notes
    -----
    Returns 0.5 if one of h1 or h2 contains only zeros.
    """
    return (1. - correlate(h1, h2))/2.


# ///////////////////////////// #
# Cross-bin comparison measures #
# ///////////////////////////// #

def quadratic_forms(h1, h2):
    r"""
    Quadrativ forms metric.
    
    Notes
    -----
    UNDER DEVELOPMENT
    
    This distance measure shows very strange behaviour. The expression
    transpose(h1-h2) * A * (h1-h2) yields egative values that can not be processed by the
    square root. Some examples::
    
        h1        h2                                          transpose(h1-h2) * A * (h1-h2)
        [1, 0] to [0.0, 1.0] :                                -2.0
        [1, 0] to [0.5, 0.5] :                                 0.0
        [1, 0] to [0.6666666666666667, 0.3333333333333333] :   0.111111111111
        [1, 0] to [0.75, 0.25] :                               0.0833333333333
        [1, 0] to [0.8, 0.2] :                                 0.06
        [1, 0] to [0.8333333333333334, 0.16666666666666666] :  0.0444444444444
        [1, 0] to [0.8571428571428572, 0.14285714285714285] :  0.0340136054422
        [1, 0] to [0.875, 0.125] :                             0.0267857142857
        [1, 0] to [0.8888888888888888, 0.1111111111111111] :   0.0216049382716
        [1, 0] to [0.9, 0.1] :                                 0.0177777777778
        [1, 0] to [1, 0]:                                      0.0
    
    It is clearly undesireable to recieve negative values and even worse to get a value
    of zero for other cases than the same histograms.
    """
    h1, h2 = __prepare_histogram(h1, h2)
    A = __quadratic_forms_matrix_euclidean(h1, h2)
    return math.sqrt((h1-h2).dot(A.dot(h1-h2))) # transpose(h1-h2) * A * (h1-h2)
    
def __quadratic_forms_matrix_euclidean(h1, h2):
    r"""
    Compute the bin-similarity matrix for the quadratic form distance measure.
    The matric :math:`A` for two histograms :math:`H` and :math:`H'` of size :math:`m` and
    :math:`n` respectively is defined as
    
    .. math::
    
        A_{m,n} = 1 - \frac{d_2(H_m, {H'}_n)}{d_{max}}
    
    with
    
    .. math::
    
       d_{max} = \max_{m,n}d_2(H_m, {H'}_n)
    
    See also
    --------
    quadratic_forms
    """
    A = scipy.repeat(h2[:,scipy.newaxis], h1.size, 1) # repeat second array to form a matrix
    A = scipy.absolute(A - h1) # euclidean distances
    return 1 - (A / float(A.max()))


# //////////////// #
# Helper functions #
# //////////////// #

def __prepare_histogram(h1, h2):
    """Convert the histograms to scipy.ndarrays if required."""
    h1 = h1 if scipy.ndarray == type(h1) else scipy.asarray(h1)
    h2 = h2 if scipy.ndarray == type(h2) else scipy.asarray(h2)
    if h1.shape != h2.shape or h1.size != h2.size:
        raise ValueError('h1 and h2 must be of same shape and size')
    return h1, h2

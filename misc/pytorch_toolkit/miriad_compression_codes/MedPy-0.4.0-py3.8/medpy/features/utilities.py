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
# version r0.1.1
# since 2013-08-24
# status Release

# build-in modules

# third-party modules
import numpy

# own modules

# code


def normalize(vector, cutoffp = (0, 100), model = False):
    r"""
    Returns a feature-wise normalized version of the supplied vector. Normalization is
    achieved to [0,1] over the complete vector using shifting and scaling.
    
    When cut-off percentile (cutoffp) values other than (0, 100) are supplied, the values
    lying before or behind the supplied percentiles are cut-off i.e. shifted to fit the
    range. 
    
    When model is set to True, an additional model describing the normalization is
    returned, that can at a later point be passed to the `normalize_with_model` function
    to normalize other feature vectors accordingly to the one passed.
    
    The vector is expected to have the form samples*features i.e.::
    
            s1    s2    s3    [...]
        f1
        f2
        [...]
    
    Therefore a supplied vector::
    
            s1    s2    s3
        f1   1.5    1    2
        f2    -1    0    1
    
    would result in the returned vector::
    
            s1    s2    s3
        f1 0.50  0.00  1.00
        f2 0.00  0.50  1.00
    
    Parameters
    ----------
    vector : sequence
        A sequence of feature vectors to normalize.
    cutoffp : (float, float)
        Cut-off percentiles.
    model : bool
        Whether to return the learned normalization model.
    
    Returns
    -------
    normalized_feature_vectors : ndarray
        The normalized versions of the input vectors.
    model : tuple, optional
        The learned normalization model.
    
    """
    vector = numpy.array(vector, dtype=numpy.float)
    
    # add a singleton dimension if required
    if 1 == vector.ndim:
        vector = vector[:, None]
    
    # compute lower and upper range border of each row using the supplied percentiles
    minp, maxp = numpy.percentile(vector, cutoffp, 0)
    
    # shift outliers to fit range
    for i in range(vector.shape[1]):
        vector[:,i][vector[:,i] < minp[i]] = minp[i]
        vector[:,i][vector[:,i] > maxp[i]] = maxp[i]
    
    # normalize
    minv = vector.min(0)
    vector -= minv
    maxv = vector.max(0)
    vector /= maxv
    
    if not model:
        return vector
    else:
        return vector, (minp, maxp, minv, maxv)
    
def normalize_with_model(vector, model):
    r"""
    Normalize as with `normalize`, but not based on the data of the passed feature
    vector, but rather on a learned model created with `normalize`. Thus formerly
    unseen query data can be normalized according to the training data.
    
    Parameters
    ----------
    vector : sequence
        A sequence of feature vectors to normalize.
    model : tuple
        A normalization model created with `normalize`.
    
    Returns
    -------
    normalize : ndarray
        The normalized versions of the input vectors.
    """
    vector = numpy.array(vector, dtype=numpy.float)
    
    # unpack model
    minp, maxp, minv, maxv = model
    
    # add a singleton dimension if required
    if 1 == vector.ndim:
        vector = vector[:, None]
    
    # shift outliers to fit range
    for i in range(vector.shape[1]):
        vector[:,i][vector[:,i] < minp[i]] = minp[i]
        vector[:,i][vector[:,i] > maxp[i]] = maxp[i]
        
    # normalize
    vector -= minv
    vector /= maxv
    
    return vector        

def append(*vectors):
    r"""
    Takes an arbitrary number of vectors containing features and append them
    (horizontally).
    
    E.g. taking a 100 and a 200 sample vector with 7 features each, a 300x7
    vector is returned.
    
    The vectors are expected to have the form samples*features i.e.::
    
            s1    s2    s3    [...]
        f1
        f2
        [...]
    
    Parameters
    ----------
    *vectors : sequences
        A number of vectors with the same number and type of features.
    
    Returns
    -------
    vector : ndarray
        The appended vectors.
    """
    # check supplied arguments
    if len(vectors) < 2:
        return vectors[0]
    
    # process supplied arguments
    vectors = list(vectors)
    for i in range(len(vectors)):
        vectors[i] = numpy.asarray(vectors[i])
        if vectors[i].ndim == 1:
            vectors[i] = numpy.asarray([vectors[i]]).T

    return numpy.squeeze(numpy.concatenate(vectors, 0))
    
def join(*vectors):
    r"""
    Takes an arbitrary number of aligned vectors of the same length and combines
    them into a single vector (vertically).
    
    E.g. taking two 100-sample feature vectors of once 5 and once 7 features, a 100x12
    feature vector is created and returned. 
    
    The feature vectors are expected to have the form samples*features i.e.::
    
            s1    s2    s3    [...]
        f1
        f2
        [...]
    
    Parameters
    ----------
    *vectors : sequences
        A number of vectors with the same number of samples.
    
    Returns
    -------
    vector : ndarray
        The combined vectors.
    """
    # check supplied arguments
    if len(vectors) < 2:
        return vectors[0]

    # process supplied arguments
    vectors = list(vectors)
    for i in range(len(vectors)):
        vectors[i] = numpy.array(vectors[i], copy=False)
        if vectors[i].ndim == 1:
            vectors[i] = numpy.array([vectors[i]], copy=False).T
    
    # treat single-value cases special (no squeezing)
    if 1 == len(vectors[0]):
        return numpy.concatenate(vectors, 1)
    
    return numpy.squeeze(numpy.concatenate(vectors, 1))

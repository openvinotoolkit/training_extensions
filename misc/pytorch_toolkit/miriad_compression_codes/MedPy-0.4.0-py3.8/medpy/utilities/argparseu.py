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
# since 2013-07-02
# status Release

# build-in modules
import argparse
import itertools
import os

# third-party modules

# own modules

# code
def existingDirectory(string):
    """
    A custom type for the argparse commandline parser.
    Check whether the supplied string points to a valid directory.
    
    Examples
    --------
    
    >>> parser.add_argument('argname', type=existingDirectory, help='help')    
    """
    if not os.path.isdir(string):
        argparse.ArgumentTypeError('{} is not a valid directory.'.format(string))
    return string

def sequenceOfStrings(string):
    """
    A custom type for the argparse commandline parser.
    Accepts colon-separated lists of strings.
    
    Examples
    --------
    
    >>> parser.add_argument('argname', type=sequenceOfStrings, help='help')
    """
    return string.split(',')

def sequenceOfIntegersGeAscendingStrict(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid integer values that are greater than or
    equal to 0 and in ascending order.
    
    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfIntegersGeAscending, help='help')
    """
    return __sequenceAscendingStrict(__sequenceGe(sequenceOfIntegers(string)))

def sequenceOfIntegers(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid integer values.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfIntegers, help='help')

    """
    value = list(map(int, string.split(',')))
    return value

def sequenceOfIntegersGt(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid integer values that are greater than 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfIntegersGt, help='help')

    """
    value = sequenceOfIntegers(string)
    return __sequenceGt(value)

def sequenceOfIntegersGe(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid integer values that are greater than or
    equal to 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfIntegersGe, help='help')

    """
    value = sequenceOfIntegers(string)
    return __sequenceGe(value)

def sequenceOfIntegersLt(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid integer values that are less than 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfIntegersLt, help='help')

    """
    value = sequenceOfIntegers(string)
    return __sequenceLt(value)

def sequenceOfIntegersLe(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid integer values that are less than or
    equal to 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfIntegersLe, help='help')

    """
    value = sequenceOfIntegers(string)
    return __sequenceLe(value)

def sequenceOfFloats(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid float values.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfFloats, help='help')

    """
    value = list(map(float, string.split(',')))
    return value

def sequenceOfFloatsGt(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid float values that are greater than 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfFloatsGt, help='help')

    """
    value = sequenceOfFloats(string)
    return __sequenceGt(value)

def sequenceOfFloatsGe(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid float values that are greater than or
    equal to 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfFloatsGe, help='help')

    """
    value = sequenceOfFloats(string)
    return __sequenceGe(value)

def sequenceOfFloatsLt(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid float values that are less than 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfFloatsLt, help='help')

    """
    value = sequenceOfFloats(string)
    return __sequenceLt(value)

def sequenceOfFloatsLe(string):
    """
    A custom type for the argparse commandline parser.
    Accepts only colon-separated lists of valid float values that are less than or
    equal to 0.

    Examples
    --------

    >>> parser.add_argument('argname', type=sequenceOfFloatsLe, help='help')

    """
    value = sequenceOfFloats(string)
    return __sequenceLe(value)

def __sequenceGt(l):
    "Test a sequences values for being greater than 0."
    for e in l:
        if 0 >= e: raise argparse.ArgumentTypeError('All values have to be greater than 0.')
    return l

def __sequenceGe(l):
    "Test a sequences values for being greater than or equal to 0."
    for e in l:
        if 0 > e: raise argparse.ArgumentTypeError('All values have to be greater than or equal to 0.')
    return l

def __sequenceLt(l):
    "Test a sequences values for being less than 0."
    for e in l:
        if 0 <= e: raise argparse.ArgumentTypeError('All values have to be less than 0.')
    return l

def __sequenceLe(l):
    "Test a sequences values for being less than or equal to 0."
    for e in l:
        if 0 < e: raise argparse.ArgumentTypeError('All values have to be less than or equal to 0.')
    return l

def __sequenceAscendingStrict(l):
    "Test a sequences values to be in strictly ascending order."
    it = iter(l)
    next(it)
    if not all(b > a for a, b in zip(l, it)):
        raise argparse.ArgumentTypeError('All values must be given in strictly ascending order.')
    return l

def __sequenceDescendingStrict(l):
    "Test a sequences values to be in strictly descending order."
    it = iter(l)
    next(it)
    if not all(b < a for a, b in zip(l, it)):
        raise argparse.ArgumentTypeError('All values must be given in strictly descending order.')
    return l
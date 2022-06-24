"""
=====================================================================
Core functionalities and shared exception objects (:mod:`medpy.core`)
=====================================================================
.. currentmodule:: medpy.core

This package collect the packages core functionalities, such as an
event Logger and shared exception classes. If you do not intend to
develop MedPy, you usually won't have to touch this.

Logger :mod:`medy.core.logger`
==============================

.. module:: medpy.core.logger
.. autosummary::
    :toctree: generated/
    
    Logger
    
 
Exceptions :mod:`medpy.core.exceptions`
=======================================

.. module:: medpy.core.exceptions
.. autosummary::
    :toctree: generated/
    
    ArgumentError
    FunctionError
    SubprocessError
    ImageLoadingError
    DependencyError
    ImageSavingError
    ImageTypeError
    MetaDataError

"""

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

# import all functions/methods/classes into the module
from .logger import Logger
from .exceptions import ArgumentError, FunctionError, SubprocessError, ImageLoadingError, \
                        DependencyError, ImageSavingError, ImageTypeError, MetaDataError
                        
# import all sub-modules in the __all__ variable
__all__ = [s for s in dir() if not s.startswith('_')]
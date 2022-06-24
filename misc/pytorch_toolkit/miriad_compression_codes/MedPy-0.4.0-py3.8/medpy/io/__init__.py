"""
===========================================
Image I/O functionalities (:mod:`medpy.io`)
===========================================
.. currentmodule:: medpy.io

This package provides functionalities for loading and saving images,
as well as the handling of image metadata.

Loading an image
================

.. module:: medpy.io.load
.. autosummary::
    :toctree: generated/
    
    load

Saving an image
===============

.. module:: medpy.io.save
.. autosummary::
    :toctree: generated/
    
    save
 
Reading / writing metadata (:mod:`medpy.io.header`)
===================================================
 
.. module:: medpy.io.header
.. autosummary::
    :toctree: generated/
    
    Header

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
from .load import load
from .save import save
from .header import \
    Header, \
    get_voxel_spacing, get_pixel_spacing, get_offset, \
    set_voxel_spacing, set_pixel_spacing, set_offset, \
    copy_meta_data

# import all sub-modules in the __all__ variable
__all__ = [s for s in dir() if not s.startswith('_')]

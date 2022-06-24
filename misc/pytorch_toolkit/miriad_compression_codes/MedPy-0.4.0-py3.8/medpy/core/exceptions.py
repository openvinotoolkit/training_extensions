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
# version r0.2
# since 2011-12-11
# status Development

# build-in module

# third-party modules

# path changes

# own modules

# code
class ArgumentError(Exception):
    r"""Thrown by an application when an invalid command line argument has been supplied.
    """
    pass
    
class FunctionError(Exception):
    r"""Thrown when a supplied function returns unexpected results.
    """
    pass
    
class SubprocessError(Exception):
    r"""Thrown by an application when a subprocess execution failed.
    """
    pass

class ImageTypeError(Exception):
    r"""Thrown when trying to load or save an image of unknown type.
    """
    pass

class DependencyError(Exception):
    r"""Thrown when a required module could not be loaded.
    """
    pass

class ImageLoadingError(Exception):
    r"""Thrown when a image could not be loaded.
    """
    pass

class ImageSavingError(Exception):
    r"""Thrown when a image could not be saved.
    """
    pass

class MetaDataError(Exception):
    r"""Thrown when an image meta data failure occurred.
    """
    pass

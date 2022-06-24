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
# version r0.1
# since 2011-12-12
# status Release

# build-in module
import sys
import logging
from logging import Logger as NativeLogger

# third-party modules

# own modules

# constants

# code
class Logger (NativeLogger):
    r"""Logger to be used by all applications and classes.
    
    Notes
    -----
    Singleton class i.e. setting the log level changes the output globally.
    
    Examples
    --------
    Initializing the logger
    
    >>> from medpy.core import Logger
    >>> logger = Logger.getInstance()
    
    Error messages are passed to stdout
    
    >>> logger.error('error message')
    15.09.2014 12:40:25 [ERROR   ] error message
    >>> logger.error('critical message')
    15.09.2014 12:40:42 [CRITICAL] critical message
    
    But debug and info messages are suppressed
    
    >>> logger.info('info message')
    >>> logger.debug('debug message')
    
    Unless the log level is set accordingly
    
    >>> import logging
    >>> logger.setLevel(logging.DEBUG)
    
    >>> logger.info('info message')
    15.09.2014 12:43:06 [INFO    ] info message (in <ipython-input-14-a08cad56519d>.<module>:1)
    >>> logger.debug('debug message')
    15.09.2014 12:42:50 [DEBUG   ] debug message (in <ipython-input-13-3bb0c512b560>.<module>:1)
        
    """
    
    class LoggerHelper (object):
        r"""A helper class which performs the actual initialization.
        """
        def __call__(self, *args, **kw) :
            # If an instance of TestSingleton does not exist,
            # create one and assign it to TestSingleton.instance.
            if Logger._instance is None :
                Logger._instance = Logger()
            # Return TestSingleton.instance, which should contain
            # a reference to the only instance of TestSingleton
            # in the system.
            return Logger._instance
    
    r"""Member variable initiating and returning the instance of the class."""    
    getInstance = LoggerHelper() 
    r"""The member variable holding the actual instance of the class."""
    _instance = None
    r"""Holds the loggers handler for format changes."""
    _handler = None

    def __init__(self, name = 'MedPyLogger', level = 0) :
        # To guarantee that no one created more than one instance of Logger:
        if not Logger._instance == None :
            raise RuntimeError('Only one instance of Logger is allowed!')
        
        # initialize parent
        NativeLogger.__init__(self, name, level)
        
        # set attributes
        self.setHandler(logging.StreamHandler(sys.stdout))
        self.setLevel(logging.WARNING)
        
    def setHandler(self, hdlr):
        r"""Replace the current handler with a new one.
        
        Parameters
        ----------
        hdlr : logging.Handler
            A subclass of Handler that should used to handle the logging output. 
        
        Notes
        -----
        If none should be replaces, but just one added, use the parent classes
        addHandler() method.
        """
        if None != self._handler:
            self.removeHandler(self._handler)
        self._handler = hdlr
        self.addHandler(self._handler)
        
    def setLevel(self, level):
        r"""Overrides the parent method to adapt the formatting string to the level.
        
        Parameters
        ----------
        level : int
            The new log level to set. See the logging levels in the logging module for details.
            
        Examples
        --------
        >>> import logging
        >>> Logger.setLevel(logging.DEBUG)
        """
        if logging.DEBUG >= level:
            formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s (in %(module)s.%(funcName)s:%(lineno)s)", 
                                          "%d.%m.%Y %H:%M:%S") 
            self._handler.setFormatter(formatter)
        else:
            formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", 
                                          "%d.%m.%Y %H:%M:%S") 
            self._handler.setFormatter(formatter)
            
        NativeLogger.setLevel(self, level)
            

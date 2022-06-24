"""
==========================================================================================
Utilities without a direct connection to medical image processing (:mod:`medpy.utilities`)
==========================================================================================
.. currentmodule:: medpy.utilities

Note that the methods/classes from the sub-modules are not loaded into
:mod:`medpy.utilities` directly, but have to be imported like

>>> from medpy.utilities.argparseu import sequenceOfIntegers

Custom types for the `argparse <https://docs.python.org/3/library/argparse.html>`_ commandline parser
=====================================================================================================

.. module:: medpy.utilities.argparseu
.. autosummary::
    :toctree: generated/
    
    sequenceOfIntegers
    sequenceOfIntegersGt
    sequenceOfIntegersGe
    sequenceOfIntegersLt
    sequenceOfIntegersLe
    sequenceOfIntegersGeAscendingStrict
    sequenceOfFloats
    sequenceOfFloatsGt
    sequenceOfFloatsGe
    sequenceOfFloatsLt
    sequenceOfFloatsLe

"""
from . import argparseu
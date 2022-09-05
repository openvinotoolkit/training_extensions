"""This module implements the ID entity."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional, Union

from bson import ObjectId


class ID(str):
    """An identifier for objects that can be persisted in repositories.

    Usually the creation of IDs is handled by the repositories. In that case objects are initialized with an empty ID()
        >>> str(ID())
        ''

    Args:
        Optional[Union[str, ObjectId]]: value of the identifier
    """

    # Instead of using composition, we directly subclass the str builtin type.
    # This allows us to reuse most of the builtin functions of the str type,
    # and avoid the overhead of Python function calls for operations such as hashing and comparison.
    # In Python 3.6, using this approach, hash and equality are about 5 times faster than using composition.
    # Since the str type is immutable, we cannot handle the object construction arguments in __init__
    # as we would normally and instead have to use __new__.
    # The __init__ function is still left for typing correctness and so that Sphinx can get the prototype of the class.

    def __new__(cls, representation: Optional[Union[str, ObjectId]] = None):
        """Creates a new ID object."""
        if representation is None:
            representation = ""
        elif isinstance(representation, ObjectId):
            representation = str(representation)
        else:
            representation = str(representation).strip().lower()

        # Ignore typing error because Mypy does not support calling super().__new__ with a parameter.
        return super().__new__(cls, representation)  # type: ignore

    # See comment block above
    # pylint: disable=W0231
    def __init__(self, representation: Optional[Union[str, ObjectId]] = None):
        pass

    # This property name is there to allow automatic mapping between ID <> IDMessage
    # It should be the same as the argument in init
    @property
    def representation(self):
        """Returns the value of the identifier."""
        return self

    def __repr__(self):
        """Returns the representation of the identifier."""
        return f"ID({self})"

"""Instantiator functions for OTX model.module components."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from functools import partial


def partial_instantiate_class(init: dict) -> partial:
    """Partially instantiates a class with the given initialization arguments.

    Copy from lightning.pytorch.cli.instantiate_class and modify it to use partial.

    Args:
        init (dict): A dictionary containing the initialization arguments.
            It should have the following keys:
            - "init_args" (dict): A dictionary of keyword arguments to be passed to the class constructor.
            - "class_path" (str): The fully qualified path of the class to be instantiated.

    Returns:
        partial: A partial object representing the partially instantiated class.
    """
    kwargs = init.get("init_args", {})
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return partial(args_class, **kwargs)

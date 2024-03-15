"""Composed dataloader."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.utils.logger import get_logger

logger = get_logger()


class CDLIterator:
    """Iterator for aligning the number of batches as many as samples in the first iterator."""

    def __init__(self, cdl):
        self._cdl = cdl
        self._index = 0
        self._cdl_iter = [iter(dl) for dl in self._cdl.loaders]

    def __next__(self):
        """Generate the next batch."""
        if self._index < self._cdl.max_iter:
            batches = {}
            for i, iterator in enumerate(self._cdl_iter):
                if i == 0:
                    batches = next(iterator)
                else:
                    try:
                        batches[f"extra_{i-1}"] = next(iterator)
                    except StopIteration:
                        self._cdl_iter[1] = iter(self._cdl.loaders[1])
                        batches[f"extra_{i-1}"] = next(self._cdl_iter[1])
            self._index += 1
            return batches
        raise StopIteration


class ComposedDL:
    """Composed dataloader for combining two or more loaders together."""

    class DummySampler:
        """Dummy sampler class to relay set_epoch() call to the list of data loaders in the CDL."""

        def __init__(self, cdl):
            self.cdl = cdl

        def set_epoch(self, epoch):
            """Set epoch."""
            loaders = self.cdl.loaders
            for loader in loaders:
                loader.sampler.set_epoch(epoch)

    def __init__(self, loaders=None):
        if loaders is None:
            loaders = []
        self.loaders = loaders
        self.max_iter = len(self.loaders[0])
        logger.info(f"possible max iterations = {self.max_iter}")
        self._sampler = ComposedDL.DummySampler(self)

    def __len__(self):
        """Return length of the first loader."""
        return self.max_iter

    def __iter__(self):
        """Iter."""
        return CDLIterator(self)

    @property
    def sampler(self):
        """Return sampler."""
        return self._sampler

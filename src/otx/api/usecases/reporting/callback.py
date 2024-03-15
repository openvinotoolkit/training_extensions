"""Callback module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


class Callback:
    """Abstract base class used to build new callbacks.

    Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def set_params(self, params):
        """Sets callback parameters."""
        # pylint: disable=W0201
        self.params = params

    def set_model(self, model):
        """Sets callback model."""
        # pylint: disable=W0201
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        """It is called on epoch begin event."""

    def on_epoch_end(self, epoch, logs=None):
        """It is called on epoch end event."""

    def on_batch_begin(self, batch, logs=None):
        """It is called on batch begin event."""

    def on_batch_end(self, batch, logs=None):
        """It is called on batch end event."""

    def on_train_begin(self, logs=None):
        """It is called on train begin event."""

    def on_train_end(self, logs=None):
        """It is called on train end event."""

    def on_train_batch_begin(self, batch, logs):
        """It is called on train batch begin event."""

    def on_train_batch_end(self, batch, logs):
        """It is called on train batch end event."""

    def on_test_begin(self, logs):
        """It is called on test begin event."""

    def on_test_end(self, logs):
        """It is called on test end event."""

    def on_test_batch_begin(self, batch, logs):
        """It is called on test batch begin event."""

    def on_test_batch_end(self, batch, logs):
        """It is called on test batch end event."""

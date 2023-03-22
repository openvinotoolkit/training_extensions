"""Test for otx.mpa.modules.optimizer.lars."""
import pytest
import torch
import torch.nn as nn

from otx.algorithms.classification.adapters.mmcls.optimizer.lars import LARS
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.mpa.test_helpers import (
    generate_random_torch_image,
    generate_toy_cnn_model,
)


class TestLARS:
    def setup(self):
        self.model = generate_toy_cnn_model()
        self.model.train()

        self.loss = nn.MSELoss()

    @e2e_pytest_unit
    def test_init(self):
        negative_lr = -0.1
        with pytest.raises(ValueError, match="Invalid learning rate: {}".format(negative_lr)):
            LARS(self.model.parameters(), lr=negative_lr)

        lr = 1e-1
        negative_momentum = -0.1
        with pytest.raises(ValueError, match="Invalid momentum value: {}".format(negative_momentum)):
            LARS(self.model.parameters(), lr=lr, momentum=negative_momentum)

        negative_wd = -0.1
        with pytest.raises(ValueError, match="Invalid weight_decay value: {}".format(negative_wd)):
            LARS(self.model.parameters(), lr=lr, weight_decay=negative_wd)

        negative_eta = -0.1
        with pytest.raises(ValueError, match="Invalid LARS coefficient value: {}".format(negative_eta)):
            LARS(self.model.parameters(), lr=lr, eta=negative_eta)

        nesterov = True
        with pytest.raises(ValueError, match="Nesterov momentum requires a momentum and zero dampening"):
            LARS(self.model.parameters(), lr=lr, nesterov=nesterov)

        LARS(self.model.parameters(), lr=lr, exclude_bn_from_weight_decay=True)

    @e2e_pytest_unit
    def test_step(self):
        selfsl_optimizer = LARS(self.model.parameters(), lr=0.1, mode="selfsl", exclude_bn_from_weight_decay=True)
        selfsl_optimizer.zero_grad()
        input_img = generate_random_torch_image(1, 3, 3, 3)
        input_img.requires_grad = True
        target = torch.randn(1, 3)

        logit = self.model(input_img)

        loss = self.loss(logit.view(1, -1), target)
        loss.backward()
        selfsl_optimizer.step()

        optimizer = LARS(self.model.parameters(), lr=0.1, momentum=0.9, exclude_bn_from_weight_decay=True)
        optimizer.zero_grad()
        input_img = generate_random_torch_image(1, 3, 3, 3)
        input_img.requires_grad = True
        target = torch.randn(1, 3)

        logit = self.model(input_img)

        loss = self.loss(logit.view(1, -1), target)
        loss.backward()
        optimizer.step()

import numpy as np
import torch

from action_recognition.loss import SoftmaxLoss, DistillationLoss, WeightedSumLoss


def _softmax(z, dim=1):
    return np.exp(z) / np.sum(np.exp(z), axis=dim)[:, np.newaxis]


def _one_hot(x, n):
    z = np.zeros((len(x), n), dtype=np.float)
    z[np.arange(z.shape[0]), x] = 1
    return z


def _log_softmax(z):
    s = np.sum(np.exp(z), axis=1)
    return (z.T - np.log(s)).T


def _kl_div(p, q):
    return np.sum(np.where((p > 0) & (q > 0), p * np.log(p / q), 0)) / p.shape[0]


def _cross_entropy(p, q):
    return - np.sum(np.where(q > 0, p * np.log(q), 0)) / p.shape[0]


class TestSoftMaxLoss:
    def test_same(self):
        loss = SoftmaxLoss()
        p = torch.tensor([
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10],
        ], dtype=torch.float)

        q = torch.tensor([1, 2, 3])

        ref = _cross_entropy(_one_hot(q.numpy(), 4), _softmax(p.numpy()))
        val = loss(outputs=p, targets=q).item()

        assert np.abs(ref - val) < 1e-4

    def test_different(self):
        loss = SoftmaxLoss()
        p = torch.tensor([
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10],
        ], dtype=torch.float)

        q = torch.tensor([1, 3, 2])

        ref = _cross_entropy(_one_hot(q.numpy(), 4), _softmax(p.numpy()))
        val = loss(outputs=p, targets=q).item()

        assert np.abs(ref - val) < 1e-4


class TestDistilationLoss:
    def test_different_output(self, mocker):
        p = torch.tensor([
            [0, 10, 3, 0],
            [1, 2, 10, 6],
            [0, 3, 4, 10]
        ], dtype=torch.float)

        q = torch.tensor([
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 10, 0]
        ], dtype=torch.float)

        teacher_mock = mocker.Mock(return_value=p)
        loss = DistillationLoss(teacher_mock)

        ref = _kl_div(_softmax(p.numpy()), _softmax(q.numpy()))
        loss_val = loss(outputs=q, inputs=[1, 1, 1]).item()

        assert np.abs(ref - loss_val) < 1e-4

    def test_same_output(self, mocker):
        p = torch.tensor([
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10]
        ], dtype=torch.float)
        teacher_mock = mocker.Mock(return_value=p)
        loss = DistillationLoss(teacher_mock)

        ref = _kl_div(_softmax(p.numpy()), _softmax(p.numpy()))
        loss_val = loss(outputs=p, inputs=[1, 1, 1])

        assert np.abs(ref - loss_val) < 1e-4


class TestWeightedSumLoss:
    def make_mock_loss(self, return_value):
        class _Loss(torch.nn.Module):
            def forward(self, outputs, **input):
                return return_value

        return _Loss()

    def test_sum(self):
        outputs = torch.zeros(1)
        l1 = self.make_mock_loss(return_value=2)
        l2 = self.make_mock_loss(return_value=3)
        loss = WeightedSumLoss()
        loss.add_loss('1', l1, 0.3)
        loss.add_loss('2', l2, 0.7)

        loss_val = loss(outputs)

        assert np.isclose(0.3 * 2 + 0.7 * 3, loss_val.item())

    def test_normazlize(self):
        outputs = torch.zeros(1)
        l1 = self.make_mock_loss(return_value=2)
        l2 = self.make_mock_loss(return_value=3)
        loss = WeightedSumLoss(normalize=True)
        loss.add_loss('1', l1, 2)
        loss.add_loss('2', l2, 1)

        loss_val = loss(outputs)

        assert np.isclose((2 / 3) * 2 + (1 / 3) * 3, loss_val.item())

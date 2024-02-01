import json
from pathlib import Path

import pytest
from otx.hpo.hpo_base import Trial


@pytest.fixture()
def good_trial_args():
    return {"trial_id": "name", "configuration": {"hp1": 1, "hp2": 1.2}, "train_environment": {"subset_ratio": 0.5}}


@pytest.fixture()
def trial(good_trial_args):
    return Trial(**good_trial_args)


def register_scores_to_trial(trial, scores=list(range(100))):  # noqa: B006, B008
    base_resource = max(trial.score.keys()) if len(trial.score) != 0 else 0
    for idx, score in enumerate(scores):
        trial.register_score(score, base_resource + idx + 1)


class TestTrial:
    def test_init(self, good_trial_args):
        Trial(**good_trial_args)

    def test_set_iteration(self, trial):
        trial.iteration = 10
        config = trial.get_train_configuration()
        assert config["configuration"]["iterations"] == 10

    @pytest.mark.parametrize("iter_val", [-10, 0])
    def test_set_negative_iteration(self, trial, iter_val):
        with pytest.raises(ValueError, match="should be positive"):
            trial.iteration = iter_val

    def test_get_train_configuration(self, good_trial_args):
        trial = Trial(**good_trial_args)
        train_config = trial.get_train_configuration()
        assert train_config["configuration"] == good_trial_args["configuration"]
        assert train_config["train_environment"] == good_trial_args["train_environment"]

    def test_get_train_configuration_without_train_env(self, good_trial_args):
        del good_trial_args["train_environment"]
        trial = Trial(**good_trial_args)
        train_config = trial.get_train_configuration()
        assert train_config["train_environment"] is None

    @pytest.mark.parametrize("score", [-10, 12.5])
    def test_register_score(self, trial, score):
        for resource in [1, 4.3, 10]:
            trial.register_score(score, resource)

    @pytest.mark.parametrize("resource", [-10, 0])
    def test_register_score_not_postive_resource(self, trial, resource):
        score = 10
        with pytest.raises(ValueError, match="should be positive"):
            trial.register_score(score, resource)

    @pytest.mark.parametrize("mode", ["min", "max"])
    @pytest.mark.parametrize("resource_limit", [None, 10, 20])
    def test_get_best_score(self, trial, mode, resource_limit):
        scores = list(range(100))
        register_scores_to_trial(trial, scores)

        if resource_limit is not None:
            scores = {i + 1: score for i, score in enumerate(scores)}
            scores = [val for key, val in scores.items() if key <= resource_limit]

        expected_score = min(scores) if mode == "min" else max(scores)

        assert expected_score == trial.get_best_score(mode, resource_limit)

    def test_get_best_score_empty_score(self, trial):
        assert trial.get_best_score() is None

    def test_get_best_score_no_trial_to_meet_condition(self, trial):
        scores = list(range(100))
        register_scores_to_trial(trial, scores)
        assert trial.get_best_score(resource_limit=0.5) is None

    def test_get_best_score_with_empty_scores(self, trial):
        assert trial.get_best_score() is None

    def test_get_best_score_with_wrong_mode_value(self, trial):
        register_scores_to_trial(trial)
        with pytest.raises(ValueError, match="mode should be"):
            trial.get_best_score(mode="wrong")

    @pytest.mark.parametrize("resource", [12, 42.12])
    def test_get_progress(self, trial, resource):
        trial.register_score(100, resource)
        assert trial.get_progress() == resource

    def test_get_progress_not_trained_at_all(self, trial):
        assert trial.get_progress() == 0

    def test_save_results(self, trial, tmp_path: Path):
        register_scores_to_trial(trial)
        save_path = tmp_path / "test"
        trial.save_results(save_path)

        with save_path.open() as f:
            result = json.load(f)

        assert result["id"] == "name"
        assert result["configuration"]["hp1"] == 1
        assert result["configuration"]["hp2"] == 1.2
        assert result["train_environment"]["subset_ratio"] == 0.5
        for key, val in result["score"].items():
            assert int(key) - 1 == val

    def test_finalize(self, trial):
        trial.iteration = 10
        trial.register_score(10, 5)
        trial.finalize()
        assert trial.is_done()

    def test_finalize_without_registered_score(self, trial):
        trial.iteration = 10
        with pytest.raises(RuntimeError, match="didn't report any score but tries to be done"):
            trial.finalize()

    def test_is_not_done(self, trial):
        trial.iteration = 10
        assert not trial.is_done()

    def test_is_done(self, trial):
        trial.iteration = 10
        trial.register_score(10, 10)
        assert trial.is_done()

    def test_is_done_iteration_not_set_yet(self, trial):
        with pytest.raises(ValueError, match="iteration isn't set yet"):
            trial.is_done()

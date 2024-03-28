# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import copy
import json
import math
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from otx.hpo import hyperband
from otx.hpo.hpo_base import TrialStatus
from otx.hpo.hyperband import AshaTrial, Bracket, HyperBand, Rung


@pytest.fixture()
def good_trial_args():
    return {"trial_id": "name", "configuration": {"hp1": 1, "hp2": 1.2}, "train_environment": {"subset_ratio": 0.5}}


@pytest.fixture()
def trial(good_trial_args):
    return AshaTrial(**good_trial_args)


@pytest.fixture()
def good_rung_args():
    return {"resource": 10, "num_required_trial": 16, "reduction_factor": 2, "rung_idx": 0}


def register_scores_to_trial(trial, scores=list(range(100))):  # noqa: B006, B008
    base_resource = max(trial.score.keys()) if len(trial.score) != 0 else 0
    for idx, score in enumerate(scores):
        trial.register_score(score, base_resource + idx + 1)


@pytest.fixture()
def rung(good_rung_args):
    return Rung(**good_rung_args)


@pytest.fixture()
def good_bracket_args():
    hp_configs = [AshaTrial(i, {"hp1": 1, "hp2": 1.2}) for i in range(100)]
    return {
        "bracket_id": 0,
        "minimum_resource": 4,
        "maximum_resource": 64,
        "hyper_parameter_configurations": hp_configs,
        "reduction_factor": 2,
        "mode": "max",
        "asynchronous_sha": True,
    }


@pytest.fixture()
def bracket(good_bracket_args):
    return Bracket(**good_bracket_args)


@pytest.fixture()
def good_hyperband_args():
    with TemporaryDirectory() as tmp_dir:
        yield {
            "search_space": {
                "hp1": {"type": "uniform", "max": 100, "min": 10},
                "hp2": {"type": "qloguniform", "max": 1000, "min": 100, "step": 2, "log_base": 10},
            },
            "save_path": tmp_dir,
            "mode": "max",
            "num_workers": 1,
            "num_full_iterations": 64,
            "full_dataset_size": 100,
            "maximum_resource": 64,
            "minimum_resource": 1,
            "reduction_factor": 4,
            "asynchronous_sha": True,
            "asynchronous_bracket": True,
            "acceptable_additional_time_ratio": 1,
        }


@pytest.fixture()
def hyper_band(good_hyperband_args):
    return HyperBand(**good_hyperband_args)


@pytest.mark.parametrize("reduction_factor", [4, 100, 4000])
def test_check_reduction_factor_value(reduction_factor):
    hyperband._check_reduction_factor_value(reduction_factor)


@pytest.mark.parametrize("reduction_factor", [-10, 1])
def test_check_reduction_factor_lesser_value(reduction_factor):
    with pytest.raises(ValueError, match="should be greater"):
        hyperband._check_reduction_factor_value(reduction_factor)


class TestAshaTrial:
    @pytest.mark.parametrize("rung_val", [0, 10])
    def teste_set_rung(self, trial, rung_val):
        trial.rung = rung_val

    @pytest.mark.parametrize("rung_val", [-10, -3])
    def test_set_negative_rung(self, trial, rung_val):
        with pytest.raises(ValueError, match="should be positive"):
            trial.rung = rung_val

    @pytest.mark.parametrize("bracket_val", [0, 10])
    def teste_set_bracket(self, trial, bracket_val):
        trial.bracket = bracket_val

    @pytest.mark.parametrize("bracket_val", [-10, -3])
    def test_set_negative_bracket(self, trial, bracket_val):
        with pytest.raises(ValueError, match="should be positive"):
            trial.bracket = bracket_val

    def test_save_results(self, trial, tmp_path: Path):
        rung_idx = 3
        trial.rung = rung_idx
        register_scores_to_trial(trial)
        save_path = tmp_path / "test"
        trial.save_results(save_path)

        with save_path.open() as f:
            result = json.load(f)
        assert result["id"] == "name"
        assert result["configuration"]["hp1"] == 1
        assert result["configuration"]["hp2"] == 1.2
        assert result["train_environment"]["subset_ratio"] == 0.5
        assert result["rung"] == rung_idx
        for key, val in result["score"].items():
            assert int(key) - 1 == val


class TestRung:
    def test_init(self, good_rung_args):
        Rung(**good_rung_args)

    @pytest.mark.parametrize("resource", [-10, 0])
    def test_init_resource_nenative(self, good_rung_args, resource):
        wrong_trial_args = good_rung_args
        wrong_trial_args["resource"] = resource
        with pytest.raises(ValueError, match="should be positive"):
            Rung(**wrong_trial_args)

    @pytest.mark.parametrize("num_required_trial", [-10, 0])
    def test_init_num_required_trial(self, good_rung_args, num_required_trial):
        wrong_trial_args = good_rung_args
        wrong_trial_args["num_required_trial"] = num_required_trial
        with pytest.raises(ValueError, match="should be positive"):
            Rung(**wrong_trial_args)

    @pytest.mark.parametrize("reduction_factor", [-10, 0, 1])
    def test_init_wrong_reduction_factor(self, good_rung_args, reduction_factor):
        wrong_trial_args = good_rung_args
        wrong_trial_args["reduction_factor"] = reduction_factor
        with pytest.raises(ValueError, match="reduction_factor should be"):
            Rung(**wrong_trial_args)

    @pytest.mark.parametrize("rung_idx", [-10, -3])
    def test_init_wrong_rung_idx(self, good_rung_args, rung_idx):
        wrong_trial_args = good_rung_args
        wrong_trial_args["rung_idx"] = rung_idx
        with pytest.raises(ValueError, match="should be positive"):
            Rung(**wrong_trial_args)

    def test_add_new_trial(self, rung, good_trial_args):
        for _ in range(rung.num_required_trial):
            trial = AshaTrial(**good_trial_args)
            rung.add_new_trial(trial)
            assert trial.rung == rung.rung_idx
            assert trial.iteration == rung.resource
            assert trial.status == TrialStatus.READY

    def test_add_too_many_trials(self, rung, good_trial_args):
        for _ in range(rung.num_required_trial):
            rung.add_new_trial(AshaTrial(**good_trial_args))
        with pytest.raises(RuntimeError, match="already sufficient trials"):
            rung.add_new_trial(AshaTrial(**good_trial_args))

    @pytest.mark.parametrize("mode", ["max", "min"])
    def test_get_best_trial(self, rung, good_trial_args, mode):
        for score in range(rung.num_required_trial):
            trial = AshaTrial(**good_trial_args)
            rung.add_new_trial(trial)
            trial.register_score(score=score, resource=trial.iteration)

        best_trial = rung.get_best_trial(mode)

        if mode == "max":
            assert best_trial.get_best_score(mode) == rung.num_required_trial - 1
        else:
            assert best_trial.get_best_score(mode) == 0

    def test_get_best_trial_with_not_started_trial(self, rung, good_trial_args):
        for score in range(rung.num_required_trial - 1):
            trial = AshaTrial(**good_trial_args)
            rung.add_new_trial(trial)
            trial.register_score(score=score, resource=trial.iteration)

        trial = AshaTrial(**good_trial_args)
        rung.add_new_trial(trial)
        best_trial = rung.get_best_trial()

        assert best_trial.get_best_score() == rung.num_required_trial - 2

    def test_get_best_trial_when_best_trial_is_undone(self, rung, good_trial_args):
        for _ in range(rung.num_required_trial - 1):
            trial = AshaTrial(**good_trial_args)
            rung.add_new_trial(trial)
            trial.register_score(score=1, resource=trial.iteration)

        trial = AshaTrial(**good_trial_args)
        rung.add_new_trial(trial)
        trial.register_score(score=100, resource=1)
        best_trial = rung.get_best_trial()

        assert best_trial.get_best_score() == 100

    def test_get_best_trial_with_no_trial(self, rung):
        best_trial = rung.get_best_trial()
        assert best_trial is None

    def test_get_best_trial_wrong_mode_val(self, rung):
        with pytest.raises(ValueError, match="mode should be"):
            rung.get_best_trial("wrong")

    def test_need_more_trials(self, rung, good_trial_args):
        for _ in range(rung.num_required_trial):
            trial = AshaTrial(**good_trial_args)
            assert rung.need_more_trials()
            rung.add_new_trial(trial)

        assert not rung.need_more_trials()

    def test_get_num_trials(self, rung, good_trial_args):
        for idx in range(rung.num_required_trial):
            trial = AshaTrial(**good_trial_args)
            rung.add_new_trial(trial)
            assert rung.get_num_trials() == idx + 1

    def test_need_more_trails(self, rung, good_trial_args):
        for i in range(1, rung.num_required_trial + 1):
            trial = AshaTrial(**good_trial_args)
            rung.add_new_trial(trial)
            if i != rung.num_required_trial:
                assert rung.need_more_trials()
            else:
                assert not rung.need_more_trials()

    def test_is_done(self, rung, good_trial_args):
        for _ in range(rung.num_required_trial - 1):
            trial = AshaTrial(**good_trial_args)
            register_scores_to_trial(trial, list(range(rung.resource)))
            rung.add_new_trial(trial)
            assert not rung.is_done()

        trial = AshaTrial(**good_trial_args)
        register_scores_to_trial(trial, list(range(rung.resource - 1)))
        rung.add_new_trial(trial)
        assert not rung.is_done()
        trial.register_score(100, rung.resource + 1)
        assert rung.is_done()

    def test_get_trial_to_promote_not_asha(self, rung, good_trial_args):
        maximum_score = 9999999
        for _ in range(rung.num_required_trial - 1):
            trial = AshaTrial(**good_trial_args)
            register_scores_to_trial(trial, list(range(rung.resource)))
            rung.add_new_trial(trial)

        assert rung.get_trial_to_promote() is None

        trial = AshaTrial(**good_trial_args)
        register_scores_to_trial(trial, [maximum_score for _ in range(rung.resource)])
        rung.add_new_trial(trial)
        assert rung.get_trial_to_promote() == trial

        num_promoteable = rung._num_required_trial // rung._reduction_factor
        for _ in range(num_promoteable - 1):
            best_trial = rung.get_trial_to_promote()
            best_trial.rung += 1
            assert rung.get_trial_to_promote(False) is not None

        best_trial = rung.get_trial_to_promote()
        best_trial.rung += 1
        assert rung.get_trial_to_promote(False) is None

    def test_get_trial_to_promote_asha(self, rung, good_trial_args):
        num_promoteable = rung._num_required_trial // rung._reduction_factor
        for _ in range(num_promoteable // rung._reduction_factor):
            for _ in range(rung._reduction_factor):
                trial = AshaTrial(**good_trial_args)
                register_scores_to_trial(trial, list(range(rung.resource)))
                rung.add_new_trial(trial)

            assert rung.get_trial_to_promote(True) is not None
            best_trial = rung.get_trial_to_promote(True)
            best_trial.rung += 1
            assert rung.get_trial_to_promote(True) is None

    def test_get_trial_to_promote_not_running(self, rung, good_trial_args):
        for _ in range(rung.num_required_trial):
            trial = AshaTrial(**good_trial_args)
            rung.add_new_trial(trial)

        for _ in range(rung.num_required_trial):
            trial = rung.get_next_trial()
            register_scores_to_trial(trial, list(range(rung.resource)))
            trial.status = TrialStatus.RUNNING

        promoted_trial = rung.get_trial_to_promote()
        assert promoted_trial is None

        trial.status = TrialStatus.STOP
        promoted_trial = rung.get_trial_to_promote()
        assert promoted_trial.status != TrialStatus.RUNNING

    def test_get_next_trial(self, rung, good_trial_args):
        trial = AshaTrial(**good_trial_args)
        rung.add_new_trial(trial)

        new_trial = rung.get_next_trial()
        assert trial == new_trial
        trial.status = TrialStatus.RUNNING

        # running trial isn't provided
        new_trial = rung.get_next_trial()
        assert new_trial is None

        # finished trial isn't provided
        register_scores_to_trial(trial, list(range(trial.iteration)))
        trial.status = TrialStatus.STOP
        new_trial = rung.get_next_trial()
        assert new_trial is None

    def test_get_next_trial_stopped_in_progress(self, rung, trial):
        rung.add_new_trial(trial)
        register_scores_to_trial(trial, list(range(trial.iteration - 1)))
        undone_trial = rung.get_next_trial()
        assert trial == undone_trial


class TestBracket:
    def test_init(self, good_bracket_args):
        Bracket(**good_bracket_args)

    def test_init_minimum_is_negative(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["minimum_resource"] = -1
        with pytest.raises(ValueError, match="should be positive"):
            Bracket(**wrong_args)

    @pytest.mark.parametrize("reduction_factor", [-10, 0, 1])
    def test_init_wrong_reduction_factor(self, good_bracket_args, reduction_factor):
        wrong_args = good_bracket_args
        wrong_args["reduction_factor"] = reduction_factor
        with pytest.raises(ValueError, match="reduction_factor should be"):
            Bracket(**wrong_args)

    def test_init_wrong_mode_val(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["mode"] = "wrong"
        with pytest.raises(ValueError, match="mode should be"):
            Bracket(**wrong_args)

    def test_init_minimum_val_is_bigger_than_maximum_val(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["minimum_resource"] = 100
        wrong_args["maximum_resource"] = 10
        with pytest.raises(ValueError, match="should be bigger"):
            Bracket(**wrong_args)

    def test_init_empty_hyper_parameter_configurations(self, good_bracket_args):
        wrong_args = good_bracket_args
        wrong_args["hyper_parameter_configurations"] = []
        with pytest.raises(ValueError, match="not enough"):
            Bracket(**wrong_args)

    def test_max_rung(self, good_bracket_args):
        bracket = Bracket(**good_bracket_args)
        expected_val = math.ceil(
            math.log(
                good_bracket_args["maximum_resource"] / good_bracket_args["minimum_resource"],
                good_bracket_args["reduction_factor"],
            ),
        )
        assert bracket.max_rung == expected_val

    def test_calcuate_max_rung_idx(self):
        minimum_resource = 1
        maximum_resource = 100
        reduction_factor = 3

        expected_val = math.ceil(math.log(maximum_resource / minimum_resource, reduction_factor))
        assert Bracket.calcuate_max_rung_idx(minimum_resource, maximum_resource, reduction_factor) == expected_val

    @pytest.mark.parametrize(
        ("minimum_resource", "maximum_resource", "reduction_factor"),
        [(-1, 100, 3), (1, -3, 3), (1, 100, -2), (10, 3, 3)],
    )
    def test_calcuate_max_rung_with_wrong_input(self, minimum_resource, maximum_resource, reduction_factor):
        with pytest.raises(ValueError):  # noqa: PT011
            Bracket.calcuate_max_rung_idx(minimum_resource, maximum_resource, reduction_factor)

    def test_release_new_trial(self, bracket):
        while True:
            new_trial = bracket.get_next_trial()
            if new_trial is None:
                break
            new_trial.status = TrialStatus.RUNNING
            assert new_trial.bracket == bracket.id
            assert new_trial.rung == 0

    def test_promote_trial_if_available_asha(self, good_bracket_args):
        reduction_factor = good_bracket_args["reduction_factor"]
        bracket = Bracket(**good_bracket_args)
        for _ in range(reduction_factor):
            trial = bracket.get_next_trial()
            assert trial.rung == 0
            trial.register_score(score=0, resource=trial.iteration)

        trial = bracket.get_next_trial()
        assert trial.rung == 1

    def test_promote_trial_if_available_sha(self, good_bracket_args):
        good_bracket_args["asynchronous_sha"] = False
        bracket = Bracket(**good_bracket_args)
        trial_arr = []
        while True:
            trial = bracket.get_next_trial()
            if trial is None:
                break

            assert trial.rung == 0
            trial.status = TrialStatus.RUNNING
            trial_arr.append(trial)

        for trial in trial_arr:
            trial.register_score(score=1, resource=trial.iteration)
            trial.status = TrialStatus.READY

        trial = bracket.get_next_trial()
        assert trial.rung == 1

    def test_get_next_trial(self, bracket):
        while not bracket.is_done():
            trial = bracket.get_next_trial()
            assert trial is not None

            trial.status = TrialStatus.RUNNING
            trial.register_score(0, trial.iteration)
            trial.status = TrialStatus.STOP

        assert bracket.is_done()

    def test_get_next_trial_if_trial_is_always_running(self, bracket):
        trial_arr = []
        while True:
            trial = bracket.get_next_trial()
            if trial is None:
                break

            trial.status = TrialStatus.RUNNING
            trial_arr.append(trial)

        for trial in trial_arr:
            trial.register_score(0, trial.iteration)

        trial = bracket.get_next_trial()
        assert trial is None

    def test_is_done(self, bracket):
        while True:
            trial = bracket.get_next_trial()
            if trial is None:
                break

            trial.register_score(score=0, resource=trial.iteration)

        assert bracket.is_done()

    @pytest.mark.parametrize("num", [1, 5, 15])
    def test_num_trial_is_not_enough(self, good_bracket_args, num):
        wrong_bracket_args = good_bracket_args
        wrong_bracket_args["hyper_parameter_configurations"] = wrong_bracket_args["hyper_parameter_configurations"][
            :num
        ]

        with pytest.raises(ValueError, match="not enough"):
            Bracket(**wrong_bracket_args)

    def test_get_best_trial(self, bracket):
        expected_score = 999999
        trial = bracket.get_next_trial()
        expected_trial_id = trial.id
        register_scores_to_trial(
            trial,
            [expected_score for _ in range(bracket._rungs[trial.rung].resource - trial.get_progress())],
        )
        while True:
            trial = bracket.get_next_trial()
            if trial is None:
                break

            register_scores_to_trial(
                trial,
                [score for score in range(bracket._rungs[trial.rung].resource - trial.get_progress())],  # noqa: C416
            )
        trial = bracket.get_best_trial()
        assert trial.get_best_score(bracket._mode) == expected_score
        assert trial.id == expected_trial_id

    def test_get_best_trial_given_absent_trial(self, bracket):
        assert bracket.get_best_trial() is None

    def test_get_best_trial_with_one_unfinished_trial(self, bracket):
        trial = bracket.get_next_trial()
        register_scores_to_trial(trial, [1])
        best_trial = bracket.get_best_trial()
        assert trial == best_trial

    def test_save_results(self, good_bracket_args, tmp_path: Path):
        trial_num = len(good_bracket_args["hyper_parameter_configurations"])
        bracket = Bracket(**good_bracket_args)
        while True:
            trial = bracket.get_next_trial()
            if trial is None:
                break

            register_scores_to_trial(
                trial,
                [score for score in range(bracket._rungs[trial.rung].resource - trial.get_progress())],  # noqa: C416
            )

        bracket.save_results(tmp_path)

        with (tmp_path / "rung_status.json").open() as f:
            result = json.load(f)

        assert result["minimum_resource"] == good_bracket_args["minimum_resource"]
        assert result["maximum_resource"] == good_bracket_args["maximum_resource"]
        assert result["reduction_factor"] == good_bracket_args["reduction_factor"]
        assert result["mode"] == good_bracket_args["mode"]
        assert result["asynchronous_sha"] == good_bracket_args["asynchronous_sha"]
        assert result["num_trials"] == trial_num
        assert len(result["rung_status"]) == bracket.max_rung + 1
        for rung_status in result["rung_status"]:
            assert rung_status["num_trial"] == rung_status["num_required_trial"]
        for i in range(trial_num):
            assert (tmp_path / f"{i}.json").exists()

    def test_print_result(self, bracket):
        while True:
            trial = bracket.get_next_trial()
            if trial is None:
                break

            register_scores_to_trial(
                trial,
                [score for score in range(bracket._rungs[trial.rung].resource - trial.get_progress())],  # noqa: C416
            )

        bracket.print_result()

    def test_print_result_without_train(self, bracket):
        bracket.print_result()

    def test_report_trial_exit_abnormally(self, bracket):
        trial = bracket.get_next_trial()
        trial.register_score(score=0, resource=trial.iteration - 0.1)
        new_trial = bracket.get_next_trial()
        assert trial.id == new_trial.id


class TestHyperBand:
    def test_init(self, good_hyperband_args):
        HyperBand(**good_hyperband_args)

    @pytest.mark.parametrize("minimum_resource", [-10, 0])
    def test_init_not_postive_maximum_resource(self, good_hyperband_args, minimum_resource):
        wrong_arg = good_hyperband_args
        wrong_arg["minimum_resource"] = minimum_resource
        with pytest.raises(ValueError):  # noqa: PT011
            HyperBand(**wrong_arg)

    @pytest.mark.parametrize("reduction_factor", [-10, 0, 1])
    def test_init_wrong_reduction_factor(self, good_hyperband_args, reduction_factor):
        wrong_arg = good_hyperband_args
        wrong_arg["reduction_factor"] = reduction_factor
        with pytest.raises(ValueError):  # noqa: PT011
            HyperBand(**wrong_arg)

    def test_init_maximum_is_same_with_minimum(self, good_hyperband_args):
        good_hyperband_args["maximum_resource"] = good_hyperband_args["minimum_resource"]
        HyperBand(**good_hyperband_args)

    def test_init_no_minimum_resource(self, good_hyperband_args):
        del good_hyperband_args["minimum_resource"]
        HyperBand(**good_hyperband_args)

    def test_init_no_maximum_resource(self, good_hyperband_args):
        del good_hyperband_args["maximum_resource"]
        HyperBand(**good_hyperband_args)

    @pytest.mark.parametrize("num", [1, 10])
    def test_make_new_hyper_parameter_configs(self, good_hyperband_args, num):
        hb = HyperBand(**good_hyperband_args)

        while True:
            trial = hb.get_next_sample()
            if trial is None:
                break
            trial.status = TrialStatus.RUNNING

            assert 10 <= trial.configuration["hp1"] <= 100
            assert 100 <= trial.configuration["hp2"] <= 1000
            assert trial.configuration["hp2"] % 2 == 0

    def test_get_next_sample(self, hyper_band):
        while True:
            trial = hyper_band.get_next_sample()
            if trial is None:
                break
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        assert hyper_band.is_done()

    def test_get_next_sample_without_train(self, hyper_band):
        while True:
            trial = hyper_band.get_next_sample()
            if trial is None:
                break
            trial.status = TrialStatus.RUNNING

        assert not hyper_band.is_done()

    def test_report_score(self, hyper_band):
        trial = hyper_band.get_next_sample()
        score = 100
        resource = 10
        hyper_band.report_score(score, resource, trial.id)
        assert trial.score[resource] == score

    def test_report_score_trial_done(self, hyper_band):
        trial = hyper_band.get_next_sample()
        hyper_band.report_score(100, 0.1, trial.id)
        hyper_band.report_score(0, 0, trial.id, done=True)
        assert trial.is_done()

    def test_get_best_config(self, hyper_band):
        max_score = 9999999
        trial = hyper_band.get_next_sample()
        expected_configuration = {"id": trial.id, "configuration": trial.configuration}
        hyper_band.report_score(score=max_score, resource=trial.iteration, trial_id=trial.id, done=False)
        hyper_band.report_score(score=max_score, resource=trial.iteration, trial_id=trial.id, done=True)
        while True:
            trial = hyper_band.get_next_sample()
            if trial is None:
                break
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        best_config = hyper_band.get_best_config()

        assert best_config == expected_configuration

    def test_get_best_config_before_train(self, hyper_band):
        best_config = hyper_band.get_best_config()
        assert best_config is None

    def test_prior_hyper_parameters(self, good_hyperband_args):
        prior1 = {"hp1": 1, "hp2": 2}
        prior2 = {"hp1": 100, "hp2": 200}
        good_hyperband_args["prior_hyper_parameters"] = [prior1, prior2]
        hyper_band = HyperBand(**good_hyperband_args)
        first_trial = hyper_band.get_next_sample()
        first_trial.status = TrialStatus.RUNNING
        second_trial = hyper_band.get_next_sample()
        second_trial.status = TrialStatus.RUNNING

        print(first_trial.configuration)
        print(second_trial.configuration)

        assert first_trial.configuration == prior1
        assert second_trial.configuration == prior2

    @pytest.mark.parametrize("num_prior_param", [10, 100, 1000])
    def test_many_prior_hyper_parameters(self, good_hyperband_args, num_prior_param):
        prior_hyper_parameters = []
        for i in range(num_prior_param):
            new_prior = {}
            for key, val in good_hyperband_args["search_space"].items():
                new_prior[key] = val["min"] + i
            prior_hyper_parameters.append(new_prior)
        good_hyperband_args["prior_hyper_parameters"] = copy.deepcopy(prior_hyper_parameters)

        hyper_band = HyperBand(**good_hyperband_args)

        i = 0
        while i < num_prior_param:
            trial = hyper_band.get_next_sample()
            if trial is None:
                break

            if not trial.score:
                assert trial.configuration == prior_hyper_parameters[i]
                i += 1
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        assert i == num_prior_param or hyper_band.is_done()

    def test_auto_config_decrease(self, good_hyperband_args):
        full_train_resource = good_hyperband_args["maximum_resource"]
        expected_time_ratio = 4
        good_hyperband_args["expected_time_ratio"] = expected_time_ratio
        hyperband = HyperBand(**good_hyperband_args)

        total_resource = 0
        while True:
            trial = hyperband.get_next_sample()
            if trial is None:
                break

            resource = ceil(trial.iteration - trial.get_progress())
            total_resource += resource

            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        maximum_resource = full_train_resource * expected_time_ratio * hyperband.acceptable_additional_time_ratio
        assert maximum_resource >= total_resource >= maximum_resource * 0.8

    def test_auto_config_increase(self, good_hyperband_args):
        full_train_resource = good_hyperband_args["maximum_resource"]
        expected_time_ratio = 100
        good_hyperband_args["expected_time_ratio"] = expected_time_ratio
        hyperband = HyperBand(**good_hyperband_args)

        total_resource = 0
        while True:
            trial = hyperband.get_next_sample()
            if trial is None:
                break

            resource = ceil(trial.iteration - trial.get_progress())
            total_resource += resource

            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        maximum_resource = full_train_resource * expected_time_ratio * hyperband.acceptable_additional_time_ratio
        assert maximum_resource >= total_resource >= maximum_resource * 0.8

    def test_asynchronous_bracket(self, hyper_band):
        bracket_id_arr = []
        while True:
            new_trial = hyper_band.get_next_sample()
            if new_trial is None:
                break

            new_trial.status = TrialStatus.RUNNING
            if new_trial.bracket not in bracket_id_arr:
                bracket_id_arr.append(new_trial.bracket)

        assert len(bracket_id_arr) > 1

    def test_synchronous_bracket(self, good_hyperband_args):
        good_hyperband_args["asynchronous_bracket"] = False
        hyper_band = HyperBand(**good_hyperband_args)
        bracket_id_arr = []
        while True:
            new_trial = hyper_band.get_next_sample()
            if new_trial is None:
                break

            new_trial.status = TrialStatus.RUNNING
            if new_trial.bracket not in bracket_id_arr:
                bracket_id_arr.append(new_trial.bracket)

        assert len(bracket_id_arr) == 1

    def test_print_result(self, hyper_band):
        while not hyper_band.is_done():
            trial = hyper_band.get_next_sample()
            if trial is None:
                break

            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        hyper_band.print_result()

    def test_print_result_without_train(self, hyper_band):
        hyper_band.print_result()

    def test_report_trial_exit_abnormally(self, hyper_band):
        trial = hyper_band.get_next_sample()
        hyper_band.report_score(score=50, resource=trial.iteration - 0.1, trial_id=trial.id, done=False)
        new_trial = hyper_band.get_next_sample()
        assert trial.id == new_trial.id

    def test_absence_minimum_resource(self, good_hyperband_args):
        del good_hyperband_args["minimum_resource"]
        hyper_band = HyperBand(**good_hyperband_args)
        first_trial = hyper_band.get_next_sample()
        assert first_trial.iteration == hyper_band.maximum_resource
        first_validation = 2
        hyper_band.report_score(score=1, resource=first_validation, trial_id=first_trial.id)

        iter_set = set()
        while not hyper_band.is_done():
            trial = hyper_band.get_next_sample()
            iter_set.add(trial.iteration)
            hyper_band.report_score(score=1, resource=trial.iteration, trial_id=trial.id)

        s_max = math.floor(
            math.log(hyper_band.maximum_resource / first_validation, good_hyperband_args["reduction_factor"]),
        )
        expected_min = hyper_band.maximum_resource * (good_hyperband_args["reduction_factor"] ** -s_max)

        assert min(iter_set) == expected_min

    @pytest.mark.parametrize("num_trial_to_estimate", [10, 30, 100])
    def test_without_maximum_resource(self, good_hyperband_args, num_trial_to_estimate):
        del good_hyperband_args["maximum_resource"]
        max_validation = 120
        hyper_band = HyperBand(**good_hyperband_args)

        trials_to_estimate = [hyper_band.get_next_sample() for _ in range(num_trial_to_estimate)]
        for trial in reversed(trials_to_estimate[1:]):
            assert trial.iteration == good_hyperband_args["minimum_resource"]
            for i in range(1, trial.iteration + 1):
                if hyper_band.report_score(score=1, resource=i, trial_id=trial.id) == TrialStatus.STOP:
                    break
            assert i == good_hyperband_args["minimum_resource"]

        first_trial = trials_to_estimate[0]
        hyper_band.report_score(score=1, resource=max_validation, trial_id=first_trial.id)
        hyper_band.report_score(score=0, resource=0, trial_id=first_trial.id, done=True)

        assert hyper_band.maximum_resource == max_validation
        assert first_trial.estimating_max_resource

    @pytest.mark.parametrize("num_trial_to_estimate", [10, 30, 100])
    def test_auto_config_decrease_without_maximum_resource(self, good_hyperband_args, num_trial_to_estimate):
        """
        Validate auto config decreases ASHA resource well without maximum_resource.
        Current auto config doesn't consider already uesd resource by unused trials to estimate.
        To align that, this test also doesn't add it to total resource.
        """
        del good_hyperband_args["maximum_resource"]
        expected_time_ratio = 4
        max_validation = 120
        total_resource = max_validation
        good_hyperband_args["expected_time_ratio"] = expected_time_ratio
        hyperband = HyperBand(**good_hyperband_args)
        trial_ids = set()

        trials_to_estimate = []
        for _ in range(num_trial_to_estimate):
            trial = hyperband.get_next_sample()
            trials_to_estimate.append(trial)

        for trial in reversed(trials_to_estimate[1:]):
            assert trial.iteration == good_hyperband_args["minimum_resource"]
            for i in range(1, trial.iteration + 1):
                if hyperband.report_score(score=1, resource=i, trial_id=trial.id) == TrialStatus.STOP:
                    break

        first_trial = trials_to_estimate[0]
        assert first_trial.estimating_max_resource

        hyperband.report_score(score=1, resource=max_validation, trial_id=first_trial.id)
        hyperband.report_score(score=0, resource=0, trial_id=first_trial.id, done=True)

        while True:
            trial = hyperband.get_next_sample()
            if trial is None:
                break
            if trial.id not in trial_ids:
                resource = trial.iteration
                trial_ids.add(trial.id)
            else:
                resource = trial.iteration - trial.get_progress()

            total_resource += resource

            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        maximum_resource = (
            hyperband.num_full_iterations * expected_time_ratio * hyperband.acceptable_additional_time_ratio
        )

        assert maximum_resource >= total_resource >= maximum_resource * 0.8

    @pytest.mark.parametrize("num_trial_to_estimate", [10, 30, 100])
    def test_auto_config_increase_without_maximum_resource(self, good_hyperband_args, num_trial_to_estimate):
        del good_hyperband_args["maximum_resource"]
        expected_time_ratio = 100
        max_validation = 120
        good_hyperband_args["expected_time_ratio"] = expected_time_ratio
        hyperband = HyperBand(**good_hyperband_args)

        trials_to_estimate = []
        for _ in range(num_trial_to_estimate):
            trial = hyperband.get_next_sample()
            trials_to_estimate.append(trial)

        for trial in reversed(trials_to_estimate[1:]):
            assert trial.iteration == good_hyperband_args["minimum_resource"]
            for i in range(1, trial.iteration + 1):
                if hyperband.report_score(score=1, resource=i, trial_id=trial.id) == TrialStatus.STOP:
                    break

        first_trial = trials_to_estimate[0]
        assert first_trial.estimating_max_resource

        hyperband.report_score(score=1, resource=max_validation, trial_id=first_trial.id)
        hyperband.report_score(score=0, resource=0, trial_id=first_trial.id, done=True)

        total_resource = max_validation
        trial_ids = set()
        while True:
            trial = hyperband.get_next_sample()
            if trial is None:
                break
            if trial.id not in trial_ids:
                resource = trial.iteration
                trial_ids.add(trial.id)
            else:
                resource = trial.iteration - trial.get_progress()

            total_resource += resource

            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyperband.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        maximum_resource = (
            hyperband.num_full_iterations * expected_time_ratio * hyperband.acceptable_additional_time_ratio
        )
        assert maximum_resource >= total_resource >= maximum_resource * 0.8

    @pytest.mark.parametrize("num_trial_to_estimate", [10, 30, 100])
    def test_without_minimum_maximum_resource(self, good_hyperband_args, num_trial_to_estimate):
        del good_hyperband_args["minimum_resource"]
        del good_hyperband_args["maximum_resource"]
        hyper_band = HyperBand(**good_hyperband_args)
        first_trial = hyper_band.get_next_sample()
        validation_interval = 2
        max_validation = 120

        trials_to_estimate = []
        for _ in range(num_trial_to_estimate):
            trial = hyper_band.get_next_sample()
            trials_to_estimate.append(trial)

        for trial in trials_to_estimate[1:]:
            for i in range(validation_interval, trial.iteration + 1, validation_interval):
                if hyper_band.report_score(score=1, resource=i, trial_id=trial.id) == TrialStatus.STOP:
                    break

        hyper_band.report_score(score=1, resource=max_validation, trial_id=first_trial.id)
        hyper_band.report_score(score=0, resource=0, trial_id=first_trial.id, done=True)

        iter_set = set()
        while not hyper_band.is_done():
            trial = hyper_band.get_next_sample()
            iter_set.add(trial.iteration)
            hyper_band.report_score(score=1, resource=trial.iteration, trial_id=trial.id)

        s_max = math.floor(
            math.log(hyper_band.maximum_resource / validation_interval, good_hyperband_args["reduction_factor"]),
        )

        expected_min = hyper_band.maximum_resource * (good_hyperband_args["reduction_factor"] ** -s_max)

        assert first_trial.estimating_max_resource
        assert min(iter_set) == expected_min
        assert hyper_band.maximum_resource == max_validation

    @pytest.mark.parametrize("expected_time_ratio", [3, 4, 5, 6])
    def test_hyperband_without_minimum_resource(self, good_hyperband_args, expected_time_ratio):
        """
        validate that when there is no minimum_resource,
        first trial stops near iteration given after ASHA schedule is made
        """
        good_hyperband_args["expected_time_ratio"] = expected_time_ratio
        del good_hyperband_args["minimum_resource"]
        hyper_band = HyperBand(**good_hyperband_args)

        val_interval = 3
        trial = hyper_band.get_next_sample()

        for i in range(val_interval, trial.iteration + 1, val_interval):
            score = i + 1
            trial_status = hyper_band.report_score(score, i, trial.id)
            if trial_status == TrialStatus.STOP:
                break

        hyper_band.report_score(score, i, trial.id, True)
        assert trial.get_progress() < trial.iteration + val_interval

    def test_get_done_progress(self, hyper_band: HyperBand):
        while not hyper_band.is_done():
            trial = hyper_band.get_next_sample()
            if trial is None:
                break

            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
            hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        assert hyper_band.get_progress() == 1

    @pytest.mark.parametrize("expected_time_ratio", [3, 4, 5, 6])
    def test_get_progress_with_expected_time_ratio(self, good_hyperband_args, expected_time_ratio):
        good_hyperband_args["expected_time_ratio"] = expected_time_ratio
        del good_hyperband_args["minimum_resource"]
        hyper_band = HyperBand(**good_hyperband_args)

        trial = hyper_band.get_next_sample()
        hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
        hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        expected_total_resource = (
            good_hyperband_args["num_full_iterations"]
            * good_hyperband_args["num_workers"]
            * good_hyperband_args["acceptable_additional_time_ratio"]
            * expected_time_ratio
        )

        assert math.isclose(hyper_band.get_progress(), trial.get_progress() / expected_total_resource)

    def test_get_progress_with_out_expected_time_ratio(self, good_hyperband_args):
        hyper_band = HyperBand(**good_hyperband_args)
        full_asha_resource = _get_full_asha_resource(
            good_hyperband_args["maximum_resource"],
            good_hyperband_args["minimum_resource"],
            good_hyperband_args["reduction_factor"],
        )

        trial = hyper_band.get_next_sample()
        hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=False)
        hyper_band.report_score(score=50, resource=trial.iteration, trial_id=trial.id, done=True)

        assert math.isclose(hyper_band.get_progress(), trial.get_progress() / full_asha_resource)


def _get_full_asha_resource(
    maximum_resource: float | int,
    minimum_resource: float | int,
    reduction_factor: int,
) -> int | float:
    total_resource: int | float = 0
    s_max = math.floor(math.log(maximum_resource / minimum_resource, reduction_factor))
    for idx in range(s_max + 1):
        num_max_rung_trials = math.floor((s_max + 1) / (idx + 1))
        total_resource += _calculate_bracket_resource(maximum_resource, reduction_factor, num_max_rung_trials, idx)

    return total_resource


def _calculate_bracket_resource(
    maximum_resource: float | int,
    reduction_factor: float | int,
    num_max_rung_trials: int,
    bracket_index: int,
) -> int | float:
    """Calculate how much resource is needed for the bracket given that resume is available."""
    num_trial = num_max_rung_trials * (reduction_factor**bracket_index)
    minimum_resource = maximum_resource * (reduction_factor**-bracket_index)

    total_resource = 0
    num_rungs = math.ceil(math.log(maximum_resource / minimum_resource, reduction_factor)) + 1
    previous_resource = 0
    resource = minimum_resource
    for _ in range(num_rungs):
        total_resource += num_trial * (resource - previous_resource)
        num_trial //= reduction_factor
        previous_resource = resource
        resource *= reduction_factor

    return total_resource

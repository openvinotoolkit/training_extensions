# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from otx.core.utils.cache import TrainerArgumentsCache


class TestTrainerArgumentsCache:
    def test_trainer_arguments_cache_init(self):
        cache = TrainerArgumentsCache(max_epochs=100, val_check_interval=0)
        assert cache.args == {"max_epochs": 100, "val_check_interval": 0}

    def test_trainer_arguments_cache_update(self):
        cache = TrainerArgumentsCache(max_epochs=100, val_check_interval=0)
        cache.update(max_epochs=1, val_check_interval=1.0)
        assert cache.args == {"max_epochs": 1, "val_check_interval": 1.0}

        cache.update(val_check_interval=None)
        assert cache.args == {"max_epochs": 1, "val_check_interval": 1.0}

    def test_trainer_arguments_cache_requires_update(self):
        cache = TrainerArgumentsCache(max_epochs=100, val_check_interval=0)
        assert cache.requires_update(max_epochs=100, val_check_interval=0)
        assert cache.requires_update(max_epochs=1, val_check_interval=1.0)

        cache = TrainerArgumentsCache(max_epochs=100, val_check_interval=0)
        cache.is_trainer_args_identical = True
        assert not cache.requires_update(max_epochs=100, val_check_interval=0)

    def test_trainer_arguments_cache_get_trainer_constructor_args(self):
        cache = TrainerArgumentsCache()
        args = cache.get_trainer_constructor_args()
        assert isinstance(args, set)
        assert "max_epochs" in args
        assert "val_check_interval" in args

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
def check_quartznet_cfg(cfg):
    # check tokenizer params
    assert "tokenizer" in cfg, "Tokenizer parameters not found."
    assert "vocab_size" in cfg["tokenizer"], "Vocab size doesn't defined."
    assert isinstance(cfg["tokenizer"]["vocab_size"], int), "Incorrect type of cfg.tokenizer.vocab_size."
    # check audio transforms
    assert "audio_transforms" in cfg, "Audio transforms doesn't defined."
    for dtype in ["train", "val"]:
        assert dtype in cfg["audio_transforms"], f"Audio transforms for {dtype} mode doesn't defined."
        assert isinstance(cfg["audio_transforms"][dtype], list), "Incorrect type of cfg.audio_transforms.train."
        for t in cfg["audio_transforms"][dtype]:
            assert "name" in t, "Incorrect format of audio transform definition."
            assert "params" in t, "Incorrect format of audio transform definition."
    # check metrics
    assert "metrics" in cfg, "Validation metrics doesn's defined."
    assert isinstance(cfg["metrics"], list), "Incorrect type of metrics config."
    for metric in cfg["metrics"]:
        assert isinstance(metric, str), "Incorrect type of metric."
    # check model
    assert "model" in cfg, "Model config doesn't defined."
    assert "dtype" in cfg["model"], "Type of model doesn't defined."
    assert isinstance(cfg["model"]["dtype"], str), "Incorrect type of cfg.model.dtype."
    assert cfg["model"]["dtype"] == "QuartzNet"
    assert "params" in cfg["model"], "Params of model doesn't defined."
    assert "cfg" in cfg["model"]["params"], "Configuration of QuartzNet network doesn't defined."
    assert isinstance(cfg["model"]["params"]["cfg"], list), "Incorrect type of config of QuartzNet network."
    for layer in cfg["model"]["params"]["cfg"]:
        assert isinstance(layer, dict), "Incorrect type of layer's config."
        for p in ["channels_out", "kernel_size", "stride", "dilation", "groups", "repeat", "n_blocks"]:
            assert p in layer, f"{p} doesn't defined."
            assert isinstance(layer[p], int), f"Incorrect type of {p}."
        assert "separable" in layer, f"separable doesn't defined."
        assert isinstance(layer["separable"], bool), f"Incorrect type of separable."
    assert "n_mels" in cfg["model"]["params"], "n_mels doesn't defined."
    assert isinstance(cfg["model"]["params"]["n_mels"], int), "Incorrect type of n_mels"
    # check optimizer
    assert "optimizer" in cfg, "Optimizer params doesn't defined."
    assert isinstance(cfg["optimizer"], dict), "Incorrect type of cfg.optimizer."
    assert "learning_rate" in cfg["optimizer"], "learning_rate params doesn't defined."
    assert isinstance(cfg["optimizer"]["learning_rate"], float), "Incorrect type of cfg.optimizer.learning_rate."
    assert "weight_decay" in cfg["optimizer"], "weight_decay params doesn't defined."
    assert isinstance(cfg["optimizer"]["weight_decay"], float), "Incorrect type of cfg.optimizer.weight_decay."
    assert "betas" in cfg["optimizer"], "betas params doesn't defined."
    assert isinstance(cfg["optimizer"]["betas"], list), "Incorrect type of cfg.optimizer.betas."
    assert "epochs" in cfg["optimizer"], "epochs params doesn't defined."
    assert isinstance(cfg["optimizer"]["epochs"], int), "Incorrect type of cfg.optimizer.epochs."
    assert "lr_scheduler" in cfg["optimizer"], "lr_scheduler params doesn't defined."
    assert isinstance(cfg["optimizer"]["lr_scheduler"], bool), "Incorrect type of cfg.optimizer.lr_scheduler."
    assert "warmup_steps" in cfg["optimizer"], "warmup_steps params doesn't defined."
    assert isinstance(cfg["optimizer"]["warmup_steps"], int), "Incorrect type of cfg.optimizer.warmup_steps."
    # check pipeline
    assert "pipeline" in cfg, "Pipeline params doesn't defined."
    assert isinstance(cfg["pipeline"], dict), "Incorrect type of cfg.pipeline."
    assert "monitor" in cfg["pipeline"], "monitor params doesn't defined."
    assert isinstance(cfg["pipeline"]["monitor"], str), "Incorrect type of cfg.monitor.monitor."
    assert "monitor_mode" in cfg["pipeline"], "monitor_mode params doesn't defined."
    assert isinstance(cfg["pipeline"]["monitor_mode"], str), "Incorrect type of cfg.monitor_mode.monitor_mode."

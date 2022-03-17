import os
from addict import Dict

# pylint: disable=protected-access,too-many-statements,unspecified-encoding

def get_default_config(data_path: str):
    cfg = {
        "trainer": {
            "lr": 0.0001,
            "forward_clip_grad_norm": True,
            "checkpoint_every": 10,
            "chkpt_dir": "gan_experiment_0",
            "batch_size": 45,
            "num_workers": 12,
            "max_epochs": 350,
            "distributed": False,
            "mel_projection_loss_cooling_epochs": 150,
            "features_l1_loss_cooling_epochs": 15,
            "discriminator_smoothing_epochs": 30
        },

        "data": {
            "training_path": data_path,
            "cmudict_path": "data/cmu_dictionary",
            "text_cleaners": ["english_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "add_noise": True,
            "add_blank": True
        },

        "model": {
            "add_blank": True,
            "extra_symbol": True,
            "encoder": {
                "embed_dims": 192,
                "num_chars": 0,
                "prenet_k": 6,
                "prenet_dims": 192,
                "dropout": 0.3,
                "n_mels": 80,
                "window_size": 4
            },
            "decoder": {
                "postnet_k": 10,
                "postnet_dims": 128,
                "n_mels": 80
            }
        }
    }
    return Dict(cfg)

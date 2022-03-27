# Speech To Text - demo package

Demo package contains simple demo to use speech to text pretrained model.

## Structure of generated package:

* package
  - `README.md`
  - `openvino_speech_to_text-0.0-py3-none-any.whl`


## Prerequisites
* Python 3.8+

## Setup Demo Package

1. Install Python (version 3.8 or higher), [setuptools](https://pypi.org/project/setuptools/), [wheel](https://pypi.org/project/wheel/).

2. Install the package in the clean environment:
```
python -m pip install openvino_speech_to_text-0.0-py3-none-any.whl
```


When the package is installed, you can import it as follows:
```
python -c "from openvino_speech_to_text import create_model"
```

> **NOTE**: On Linux and macOS, you may need to type `python3` instead of `python`. You may also need to [install pip](https://pip.pypa.io/en/stable/installation/).
> For example, on Ubuntu execute the following command to get pip installed: `sudo apt install python3-pip`.

## Usecases

1. Running the `demo.py` application with the `-h` option yields the following usage message:
   ```
   usage: python -m openvino_speech_to_text.demo [-h] -i INPUT

   Options:
        usage: demo.py [-h] [-m MODEL] -i INPUT [-v VOCAB] [-d DEVICE]

        optional arguments:
          -h, --help               Show this help message and exit.
          -m MODEL, --model MODEL  Optional. Path to an .xml file with a trained model.
          -i INPUT, --input INPUT  Required. Path to an audio file in WAV PCM 16 kHz mono format
          -v VOCAB, --vocab VOCAB  Optional. Path to vocabulary file in .json format
          -d DEVICE, --device DEVICE
                                   Optional. Specify the target device to infer on, for example: CPU, GPU, HDDL, MYRIAD or HETERO.
                                   The demo will look for a suitable IE plugin for this device. Default value is CPU.

   ```

2. You can create your own demo application, using `openvino_speech_to_text`:
   ```python
   import wave
   import numpy as np
   from openvino_speech_to_text import create_model

   def load_audio_wav(audio_path: str) -> typing.Tuple[np.array, int]:
        """
        Load audio file in .wav format

        Arguments:
            audio_path (str): Path to audio file in .wav format.

        Returns:
            audio (np.array): Waveform.
            sampling_rate (int): Sampling rate.
        """
        with wave.open(audio_path, 'rb') as wave_read:
            channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
            assert sample_width == 2, "Only 16-bit WAV PCM supported"
            assert compression_type == 'NONE', "Only linear PCM WAV files supported"
            assert channel_num == 1, "Only mono WAV PCM supported"
            assert sampling_rate == 16000, "Only 16 KHz audio supported"
            audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((pcm_length, channel_num))
        return audio.flatten(), sampling_rate


   # create speech to text model
   model = create_model()
   # load audio
   audio, sampling_rate = load_audio_wav(args.input)
   # inference
   text = model(audio, sampling_rate)
   # print transcription
   print(text)
   ```
import atheris


class FuzzingHelper:
    """Helper to make required data from input_bytes for the fuzzing tests"""

    def __init__(self, input_bytes):
        """Init"""
        self.provider = atheris.FuzzedDataProvider(input_bytes)

    def get_string(self, byte_conut=256):
        """Consume a string"""
        return self.provider.ConsumeString(byte_conut)

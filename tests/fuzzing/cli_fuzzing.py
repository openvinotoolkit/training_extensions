import sys

import atheris
from helper import FuzzingHelper

from otx.cli.tools.cli import main as cli_main
from otx.cli.utils.errors import CliException


@atheris.instrument_func
def fuzz_otx(input_bytes):
    # create a FuzzingHelper instance to get suitable data type from the randomly generated 'input_bytes'
    helper = FuzzingHelper(input_bytes)
    backup_argv = sys.argv

    # get 'operation' arguments from 'input_bytes'
    operation = helper.get_string()
    sys.argv = ["otx", operation]
    try:
        _ = cli_main()
    except SystemExit as e:
        # argparser will throw SystemExit with code 2 when some required arguments are missing
        if e.code != 2:
            raise
    except CliException:
        pass
    # some known exceptions can be catched here
    finally:
        sys.argv = backup_argv


def main():
    # 'sys.argv' used to passing options to atheris.Setup()
    # available options can be found https://llvm.org/docs/LibFuzzer.html#options
    atheris.Setup(sys.argv, fuzz_otx)
    # Fuzz() will
    atheris.Fuzz()


if __name__ == "__main__":
    main()

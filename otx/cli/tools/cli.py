"""OTX CLI entry point."""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import sys

from otx.cli.utils import telemetry

from .build import main as otx_build
from .demo import main as otx_demo
from .deploy import main as otx_deploy
from .eval import main as otx_eval
from .explain import main as otx_explain
from .export import main as otx_export
from .find import main as otx_find
from .optimize import main as otx_optimize
from .train import main as otx_train

__all__ = [
    "otx_demo",
    "otx_deploy",
    "otx_eval",
    "otx_explain",
    "otx_export",
    "otx_find",
    "otx_train",
    "otx_optimize",
    "otx_build",
]


def parse_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("operation", choices=[x[4:] for x in __all__], type=str)

    return parser.parse_known_args()[0]


def main():
    """Entry point for OTX CLI.

    This function is a single entry point for all OTX CLI related operations:
      - build
      - demo
      - deploy
      - eval
      - explain
      - export
      - find
      - train
      - optimize
    """

    name = parse_args().operation
    sys.argv[0] = f"otx {name}"

    del sys.argv[1]

    tm_session = telemetry.init_telemetry_session()
    results = {}
    try:
        results = globals()[f"otx_{name}"]()
        if results is None:
            results = dict(retcode=0)
    except Exception as error:
        results["retcode"] = -1
        results["exception"] = repr(error)
        telemetry.send_cmd_results(tm_session, name, results)
        raise
    else:
        telemetry.send_cmd_results(tm_session, name, results)
    finally:
        telemetry.close_telemetry_session(tm_session)

    return results.get("retcode", 0)


if __name__ == "__main__":
    main()

from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict

from otx.cli.utils.jsonargparse import get_short_docstring
from otx.engine import BaseEngine


class CLI:
    """CLI.

    Limited CLI to show how the api does not change externally while retaining the ability to expose models from the
    adapters.
    """

    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument(
            "--config",
            action=ActionConfigFile,
            help="Configuration file in JSON format.",
        )
        self.add_subcommands()
        self.run()

    def subcommands(self):
        return ["train", "test"]

    def _get_model_classes(self):
        classes = [engine.BASE_MODEL for engine in BaseEngine.__subclasses__()]
        print(classes)
        return tuple(classes)

    def add_subcommands(self):
        parser_subcommand = self.parser.add_subcommands()
        for subcommand in self.subcommands():
            subparser = ArgumentParser()
            subparser.add_method_arguments(BaseEngine, subcommand, skip={"model"})
            subparser.add_subclass_arguments(self._get_model_classes(), "model", required=False, fail_untyped=True)
            fn = getattr(BaseEngine, subcommand)
            description = get_short_docstring(fn)
            parser_subcommand.add_subcommand(subcommand, subparser, help=description)

    def run(self):
        args = self.parser.parse_args()
        args_dict = namespace_to_dict(args)
        engine = BaseEngine()
        # do something here


if __name__ == "__main__":
    CLI()

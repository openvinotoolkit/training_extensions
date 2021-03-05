from argparse import ArgumentParser, Namespace

from action_recognition.options import get_argument_parser, BoolFlagAction
from action_recognition.utils import generate_args


class TestBoolFlag:
    def test_default_value(self):
        argparse = ArgumentParser()
        argparse.add_argument("--opt", action=BoolFlagAction)
        args = argparse.parse_args([])
        assert args.opt is False

        argparse = ArgumentParser()
        argparse.add_argument("--opt", default=True, action=BoolFlagAction)
        args = argparse.parse_args([])
        assert args.opt is True

    def test_no_prefix(self):
        argparse = ArgumentParser()
        argparse.add_argument("--opt", action=BoolFlagAction)

        args = argparse.parse_args(['--opt'])
        assert args.opt is True
        args = argparse.parse_args(['--no-opt'])
        assert args.opt is False

        argparse = ArgumentParser()
        argparse.add_argument("--opt", default=True, action=BoolFlagAction)
        args = argparse.parse_args(['--opt'])
        assert args.opt is True
        args = argparse.parse_args(['--no-opt'])
        assert args.opt is False


class TestGenerateArgs:
    def test_returns_namespace(self):
        args, _ = generate_args()

        assert isinstance(args, Namespace)

    def test_kwargs_is_setting_args(self):
        args, _ = generate_args(encoder='test')

        assert args.encoder == 'test'

    def test_bool_flags(self):
        args, _ = generate_args('--no-val')

        assert args.val is False

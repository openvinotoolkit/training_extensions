# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from jsonargparse import Namespace
from otx.v2.cli.utils.arg_parser import OTXArgumentParser, get_short_docstring, tuple_constructor
from pytest_mock.plugin import MockerFixture


def test_tuple_constructor() -> None:
    loader = yaml.Loader("")
    node = None
    assert tuple_constructor(loader, node) is None

    node = yaml.SequenceNode(tag='tag:yaml.org,2002:python/tuple', value=[yaml.ScalarNode(tag='tag:yaml.org,2002:int', value='1'), yaml.ScalarNode(tag='tag:yaml.org,2002:int', value='5')])
    assert tuple_constructor(loader, node) == (1, 5)

def test_get_short_docstring() -> None:
    class TestClass1:
        def __init__(self, arg1: str, arg2: int) -> None:
            self.arg1 = arg1
            self.arg2 = arg2

    class TestClass2:
        """Test Summary."""

        def __init__(self, arg1: str, arg2: int) -> None:
            self.arg1 = arg1
            self.arg2 = arg2

    assert get_short_docstring(TestClass1) is None
    assert get_short_docstring(TestClass2) == "Test Summary."


class TestOTXArgumentParser:
    def test_init(self) -> None:
        parser = OTXArgumentParser()
        assert isinstance(parser, OTXArgumentParser)


    def test_add_core_class_args(self, mocker: MockerFixture) -> None:
        class TestClass:
            def __init__(self, arg1: str, arg2: int) -> None:
                self.arg1 = arg1
                self.arg2 = arg2

        def mock_function(arg1: str, arg2: int) -> tuple:
            return arg1, arg2

        parser = OTXArgumentParser()
        parser.add_core_class_args(TestClass, "test")
        args = parser.parse_args(["--test.arg1", "value1", "--test.arg2", "2"])
        assert isinstance(args.test, Namespace)
        assert args.test.arg1 == "value1"
        assert args.test.arg2 == 2

        mock_add_subclass_arguments = mocker.patch("otx.v2.cli.utils.arg_parser.ArgumentParser.add_subclass_arguments")
        parser.add_core_class_args(TestClass, "test", subclass_mode=True)
        mock_add_subclass_arguments.assert_called_once()

        mock_add_dataclass_arguments = mocker.patch("otx.v2.cli.utils.arg_parser.ArgumentParser.add_dataclass_arguments")
        parser.add_core_class_args(TestClass, "test", dataclass_mode=True)
        mock_add_dataclass_arguments.assert_called_once()

        mock_class_from_function = mocker.patch("otx.v2.cli.utils.arg_parser.class_from_function")
        with pytest.raises(NotImplementedError):
            parser.add_core_class_args(api_class=mock_function, nested_key="test")
        mock_class_from_function.assert_called_once_with(mock_function)

        with pytest.raises(NotImplementedError):
            parser.add_core_class_args("test", "test")

    def test_check_config(self) -> None:
        parser = OTXArgumentParser()
        args = parser.parse_args([])
        parser.check_config(args)

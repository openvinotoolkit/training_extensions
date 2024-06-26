from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
from otx.utils.utils import (
    check_pickleable,
    find_file_recursively,
    find_unpickleable_obj,
    get_class_initial_arguments,
    get_decimal_point,
    get_using_dot_delimited_key,
    remove_matched_files,
    set_using_dot_delimited_key,
)


@pytest.fixture()
def fake_obj(mocker):
    target = mocker.MagicMock()
    target.a.b.c = {"d": mocker.MagicMock()}
    target.a.b.c["d"].e = [0, 1, 2]
    return target


def test_get_using_dot_delimited_key(fake_obj):
    assert get_using_dot_delimited_key("a.b.c.d.e.2", fake_obj) == 2


def test_set_using_dot_delimited_key(fake_obj):
    expected_val = 2
    set_using_dot_delimited_key("a.b.c.d.e.0", expected_val, fake_obj)
    assert fake_obj.a.b.c["d"].e[0] == expected_val


@pytest.mark.parametrize(("val", "decimal_point"), [(0.001, 3), (-0.0001, 4), (1, 0), (100, 0), (-2, 0)])
def test_get_decimal_point(val, decimal_point):
    assert get_decimal_point(val) == decimal_point


def test_find_file_recursively(tmp_path):
    file_name = "some_file.txt"
    target = tmp_path / "foo" / "bar" / file_name
    target.parent.mkdir(parents=True)
    target.touch()

    assert find_file_recursively(tmp_path, file_name) == target


def test_find_file_recursively_multiple_files_exist(tmp_path):
    file_name = "some_file.txt"

    target1 = tmp_path / "foo" / file_name
    target1.parent.mkdir(parents=True)
    target1.touch()

    target2 = tmp_path / "foo" / "bar" / file_name
    target2.parent.mkdir(parents=True)
    target2.touch()

    assert find_file_recursively(tmp_path, file_name) in [target1, target2]


def test_find_file_recursively_not_exist(tmp_path):
    file_name = "some_file.txt"
    assert find_file_recursively(tmp_path, file_name) is None


def make_dir_and_file(dir_path: Path, file_path: str | Path) -> Path:
    file = dir_path / file_path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch()

    return file


@pytest.fixture()
def temporary_dir_w_some_txt(tmp_path):
    some_txt = ["a/b/c/d.txt", "1/2/3/4.txt", "e.txt", "f/g.txt", "5/6/7.txt"]
    for file_path in some_txt:
        make_dir_and_file(tmp_path, file_path)
    return tmp_path


def test_remove_matched_files(temporary_dir_w_some_txt):
    file_path_to_leave = "foo/bar/file_to_leave.txt"
    file_to_leave = make_dir_and_file(temporary_dir_w_some_txt, file_path_to_leave)

    remove_matched_files(temporary_dir_w_some_txt, "*.txt", file_to_leave)

    assert file_to_leave.exists()
    assert len(list(temporary_dir_w_some_txt.rglob("*.txt"))) == 1


def test_remove_matched_files_remove_all(temporary_dir_w_some_txt):
    remove_matched_files(temporary_dir_w_some_txt, "*.txt")

    assert len(list(temporary_dir_w_some_txt.rglob("*.txt"))) == 0


def test_remove_matched_files_no_file_to_remove(temporary_dir_w_some_txt):
    remove_matched_files(temporary_dir_w_some_txt, "*.log")

    assert len(list(temporary_dir_w_some_txt.rglob("*.txt"))) == 5


def test_get_class_initial_arguments():
    class FakeCls:
        def __init__(self, a, b):
            self.init_args = get_class_initial_arguments()

    fake_cls = FakeCls(4, 5)

    assert fake_cls.init_args == (4, 5)


class FakeClassForPickleTest:
    def __init__(self):
        self.attr1 = 1
        self.attr2 = 2

    def func1():
        return 3


def fake_func_for_pickle_test(arg1):
    return arg1


def test_find_unpickleable_obj():
    mock_class = FakeClassForPickleTest()
    mock_class.attr1 = lambda: 1
    test_dict = {
        "dict_type": {
            "unpickleable_obj1": lambda: 1,
            "pickleable_obj": 1,
        },
        "list_type": [lambda: 1, 3],
        "class_type": mock_class,
        "lambda_func": lambda: 1,
        "partial_func": partial(fake_func_for_pickle_test, arg1=lambda: 1),
    }
    test_dict["dict_type"]["unpickleable_obj2"] = test_dict["dict_type"]["unpickleable_obj1"]  # set same obj
    ret = find_unpickleable_obj(test_dict, "test_dict")

    assert 'test_dict["dict_type"]["unpickleable_obj1"]' in ret
    assert 'test_dict["list_type"][0]' in ret
    assert 'test_dict["class_type"].attr1' in ret
    assert 'test_dict["lambda_func"]' in ret
    assert 'test_dict["partial_func"].keywords["arg1"]' in ret


def test_check_pickleable():
    pickleable_obj = [1, 2, 3]
    unpickleable_obj = lambda: 1
    assert check_pickleable(pickleable_obj) is True
    assert check_pickleable(unpickleable_obj) is False

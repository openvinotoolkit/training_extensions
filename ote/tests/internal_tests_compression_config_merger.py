# Copyright (C) 2020 Intel Corporation
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

import unittest

try:
    from ote.modules.compression.merger import merge_dicts_and_lists_b_into_a
except RuntimeError:
    pass


class TestCaseMergeDictsAndLists(unittest.TestCase):
    problem = None
    model = None
    topic = 'internal'

    def test_case_01(self):
        a = {"a": 1, "b": 2, "c": {"c1": "x", "c2":"y"}}
        b = {"a": 2, "c": {"c2": "z"}}
        res = merge_dicts_and_lists_b_into_a(a, b)
        self.assertEqual(res, {"a": 2, "b": 2, "c": {"c1": "x", "c2":"z"}})
    def test_case_02(self):
        a = {"a": 1, "b": 2, "c": {"c1": "x", "c2":"y"}}
        b = {"a": 2, "c": {"c2": {'p':'q'}}}
        with self.assertRaises(AssertionError):
            merge_dicts_and_lists_b_into_a(a, b)
    def test_case_03(self):
        a = {"a": 1, "b": 2, "c": {"c1": "x", "c2":"y"}}
        b = {"a": 2, "c": {"c2": [1, 2]}}
        with self.assertRaises(AssertionError):
            merge_dicts_and_lists_b_into_a(a, b)
    def test_case_04(self):
        a = {"a": 1, "b": 2, "c": {"c1": "x", "c2":['aa', 'bb', 'cc']}}
        b = {"a": 2, "c": {"c2": ['dd']}}
        res = merge_dicts_and_lists_b_into_a(a, b)
        self.assertEqual(res, {"a": 2, "b": 2, "c": {"c1": "x", "c2":['aa', 'bb', 'cc', 'dd']}})
    def test_case_05(self):
        a = {"a": 1, "b": 2, "c": {"c1": "x", "c2":['aa', 'bb', 'cc']}}
        b = {"a": 2, "cc": {"c2": ['dd']}}
        res = merge_dicts_and_lists_b_into_a(a, b)
        self.assertEqual(res, {"a": 2, "b": 2, "c": {"c1": "x", "c2":['aa', 'bb', 'cc', ]}, 'cc': {'c2': ['dd']}})
    def test_case_06(self):
        a = {"a": 1, "b": 2, "c": {"c1": "x", "c2":['aa', 'bb', 'cc']}}
        b = {"a": 2, "c": {"c3": ['dd']}}
        res = merge_dicts_and_lists_b_into_a(a, b)
        self.assertEqual(res, {"a": 2, "b": 2, "c": {"c1": "x", "c2":['aa', 'bb', 'cc', ], 'c3': ['dd']}})
    def test_case_07(self):
        a = {"a": 1, "b": 2, "c": {"c1": {'p':'q'}, "c2":['aa', 'bb', 'cc']}}
        b = {"a": 2, "c": {'c1': {'p':'q1', 'ppp':'qqq'}, "c2": ['dd']}}
        res = merge_dicts_and_lists_b_into_a(a, b)
        self.assertEqual(res, {"a": 2, "b": 2, "c": {"c1": {'p':'q1', 'ppp':'qqq'}, "c2":['aa', 'bb', 'cc', 'dd'], }})
    def test_case_08(self):
        a = {"a": 1, "b": 2, "c": {"c1": {'p':{'q':1}}, "c2":['aa', 'bb', 'cc']}}
        b = {"a": 2, "c": {'c1': {'p':'q1', 'ppp':'qqq'}, "c2": ['dd']}}
        try:
            merge_dicts_and_lists_b_into_a(a, b)
        except AssertionError as e:
            err_args = e.args
        self.assertIn('key=`c.c1.p`', err_args[0])

    def test_smoke_test(self):
        pass

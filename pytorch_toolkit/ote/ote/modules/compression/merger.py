from copy import copy

def merge_dicts_and_lists_b_into_a(a, b):
    return _merge_dicts_and_lists_b_into_a(a, b, "")

def _merge_dicts_and_lists_b_into_a(a, b, cur_key=None):
    """The function is inspired by mmcf.Config._merge_a_into_b,
    but it
    * works with usual dicts and lists and derived types
    * supports merging of lists (by concatenating the lists)
    * makes recursive merging for dict + dict case
    * overwrites when merging scalar into scalar

    Note that we merge b into a (whereas Config makes merge a into b),
    since otherwise the order of list merging is counter-intuitive.
    """
    def _err_str(_a, _b, _key):
        if _key is None:
            _key_str = 'of whole structures'
        else:
            _key_str = f'during merging for key=`{_key}`'
        return (f'Error in merging parts of config: different types {_key_str},'
                f' type(a) = {type(_a)},'
                f' type(b) = {type(_b)}')

    assert isinstance(a, (dict, list)), f'Can merge only dicts and lists, whereas type(a)={type(a)}'
    assert isinstance(a, list) == isinstance(b, list), _err_str(a, b, cur_key)
    if isinstance(a, list):
        # the main diff w.r.t. mmcf.Config -- merging of lists
        return a + b

    a = copy(a)
    for k in b.keys():
        if k not in a:
            a[k] = copy(b[k])
            continue
        new_cur_key = cur_key + '.' + k if cur_key else k
        if isinstance(a[k], (dict, list)):
            a[k] = _merge_dicts_and_lists_b_into_a(a[k], b[k], new_cur_key)
            continue

        assert not isinstance(b[k], (dict, list)), _err_str(a[k], b[k], new_cur_key)

        # suppose here that a[k] and b[k] are scalars, just overwrite
        a[k] = b[k]
    return a

SHOULD_DEBUG_MERGING_DICTS_AND_LISTS = False
if __name__ == '__main__' and SHOULD_DEBUG_MERGING_DICTS_AND_LISTS:
    import unittest
    class TestCaseMergeDictsAndLists(unittest.TestCase):
        # pylint: disable=line-too-long
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

    unittest.main(defaultTest='TestCaseMergeDictsAndLists')

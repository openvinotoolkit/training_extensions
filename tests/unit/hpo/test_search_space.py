import math
import pytest
import copy

from hpopt.search_space import SingleSearchSpace, SearchSpace

ALL_TYPE = ["uniform", "loguniform", "quniform", "qloguniform", "choice"]
NOT_CATEGORICAL_TYPE = ["uniform", "loguniform", "quniform", "qloguniform"]
USE_LOG_SCALE_TYPE = ["loguniform", "qloguniform"]
USE_QUANTIZED_STEP_TYPE = ["quniform", "qloguniform"]

def make_single_search_space_uniform_good_arguments():
    return [
        {"type" : "uniform", "min" : 1, "max" : 100},
        {"type" : "uniform", "min" : 1.12, "max" : 53.221},
        {"type" : "uniform", "min" : -300, "max" : -299},
        {"type" : "uniform", "min" : -12.543, "max" : 15.233},
    ]

def add_step_from_args(args, type_name=None):
    new_args = []
    for arg in args:
        step = (arg["max"] - arg["min"]) / 2
        arg["step"] = step
        if type_name is not None:
            arg["type"] = type_name
        new_args.append(arg)

    return new_args

def add_log_base_from_args(args, type_name):
    new_args = []
    for log_base in [2, 10]:
        for arg in args:
            if arg["min"] <= 0:
                continue
            arg["log_base"] = log_base
            if type_name is not None:
                arg["type"] = type_name
            new_args.append(arg)

    return new_args
    
def make_single_search_space_quniform_good_arguments():
    return add_step_from_args(
        make_single_search_space_uniform_good_arguments(),
        "quniform"
    )

def make_single_search_space_loguniform_good_arguments():
    return add_log_base_from_args(
        make_single_search_space_uniform_good_arguments(),
        "loguniform"
    )
        
def make_single_search_space_qloguniform_good_arguments():
    return add_log_base_from_args(
        make_single_search_space_quniform_good_arguments(),
        "qloguniform"
    )

def make_single_search_space_choice_good_arguments():
    return [
        {"type" : "choice", "choice_list" : ("abc", "def")},
        {"type" : "choice", "choice_list" : [1,2,3]}
    ]

MAKE_GOOD_ARGS = {
    "uniform" : make_single_search_space_uniform_good_arguments,
    "quniform" : make_single_search_space_quniform_good_arguments,
    "loguniform" : make_single_search_space_loguniform_good_arguments,
    "qloguniform" : make_single_search_space_qloguniform_good_arguments,
    "choice" : make_single_search_space_choice_good_arguments
}

def get_wrong_arg(original_arg, attr_names, values, errors):
    wrong_arg = copy.deepcopy(original_arg)
    if not isinstance(attr_names, (list, tuple)):
        attr_names = [attr_names]
        values = [values]

    for attr_name, value in zip(attr_names, values):
        wrong_arg[attr_name] = value
    wrong_arg["error"] = errors

    return wrong_arg

def make_arg_minmax_wrong(arg):
    args = []
    # value is None
    for attr_name in ["min", "max"]:
        args.append(get_wrong_arg(arg, attr_name, None, (ValueError, TypeError)))

    # min is greater or same than max
    for val in [(3, 1), (5.5, 2.3), (1, 1), (2.0, 2.0)]:
        args.append(get_wrong_arg(arg, ["min", "max"], val, ValueError))

    # value is minus although using log scale
    if "log_base" in arg:
        for val in [(-20, -12), (-12.124, 10), (0, 3)]:
            args.append(get_wrong_arg(arg, ["min", "max"], val, ValueError))

    return args

def make_arg_step_wrong(arg):
    args = []

    # vlaue is None
    args.append(get_wrong_arg(arg, "step", None, (TypeError, ValueError)))

    # step is too big
    args.append(
        get_wrong_arg(arg, "step", arg["max"] - arg["min"] + 1, (TypeError, ValueError))
    )

    return args

def make_arg_logbase_wrong(arg):
    args = []

    # too small value
    for val in [1, 0, -1]:
        args.append(get_wrong_arg(arg, "log_base", val, ValueError))
        
    return args

def make_arg_choicelist_wrong(arg):
    args = []

    # vlaue is None
    args.append(get_wrong_arg(arg, "choice_list", None, (TypeError, ValueError)))
    # few elements

    for val in [[], [1]]:
        args.append(get_wrong_arg(arg, "choice_list", val, ValueError))

    return args

def make_arg_type_wrong(arg):
    args = []

    # wrong type
    for val in ["wrong_type", 12, 1.24, [1,2]]:
        args.append(get_wrong_arg(arg, "type", val, ValueError))

    return args

class TestSingleSearchSpace:
    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_init_with_good_input(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            search_space_class = SingleSearchSpace(**arg)

    @pytest.mark.parametrize("type", NOT_CATEGORICAL_TYPE)
    def test_init_wrong_minmax_arg(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            wrong_args = make_arg_minmax_wrong(arg)
            for wrong_arg in wrong_args:
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                with pytest.raises(errors):
                    search_space_class = SingleSearchSpace(**wrong_arg)

    @pytest.mark.parametrize("type", USE_QUANTIZED_STEP_TYPE)
    def test_init_wrong_step_arg(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            wrong_args = make_arg_step_wrong(arg)
            for wrong_arg in wrong_args:
                print(wrong_arg)
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                with pytest.raises(errors):
                    search_space_class = SingleSearchSpace(**wrong_arg)

    @pytest.mark.parametrize("type", USE_LOG_SCALE_TYPE)
    def test_init_wrong_log_base_arg(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            wrong_args = make_arg_logbase_wrong(arg)
            for wrong_arg in wrong_args:
                print(wrong_arg)
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                with pytest.raises(errors):
                    search_space_class = SingleSearchSpace(**wrong_arg)

    def test_init_wrong_choice_list_arg(self):
        args = MAKE_GOOD_ARGS["choice"]()
        for arg in args:
            wrong_args = make_arg_choicelist_wrong(arg)
            for wrong_arg in wrong_args:
                print(wrong_arg)
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                with pytest.raises(errors):
                    search_space_class = SingleSearchSpace(**wrong_arg)

    def test_init_with_wrong_type(self):
        for type in ALL_TYPE:
            args = MAKE_GOOD_ARGS[type]()
            for arg in args:
                wrong_args = make_arg_type_wrong(arg)
                for wrong_arg in wrong_args:
                    errors = wrong_arg["error"]
                    del wrong_arg["error"]
                    with pytest.raises(errors):
                        search_space_class = SingleSearchSpace(**wrong_arg)

    def test_set_value_normally(self):
        args = []
        for type in ALL_TYPE:
            args.extend(MAKE_GOOD_ARGS[type]())
        cur_arg = args.pop(0)
        for new_arg in args:
            sss = SingleSearchSpace(**cur_arg)
            sss.set_value(**new_arg)
            cur_arg = new_arg

    @pytest.mark.parametrize("type", NOT_CATEGORICAL_TYPE)
    def test_set_value_wrong_minmax_arg(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            wrong_args = make_arg_minmax_wrong(arg)
            for wrong_arg in wrong_args:
                if wrong_arg["min"] is None or wrong_arg["max"] is None:
                    continue
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                print(wrong_arg)
                with pytest.raises(errors):
                    sss.set_value(**wrong_arg)

    @pytest.mark.parametrize("type", USE_QUANTIZED_STEP_TYPE)
    def test_set_value_wrong_step_arg(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            wrong_args = make_arg_step_wrong(arg)
            for wrong_arg in wrong_args:
                if wrong_arg["step"] is None:
                    continue
                print(wrong_arg)
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                with pytest.raises(errors):
                    sss.set_value(**wrong_arg)

    @pytest.mark.parametrize("type", USE_LOG_SCALE_TYPE)
    def test_set_value_wrong_log_base_arg(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            wrong_args = make_arg_logbase_wrong(arg)
            for wrong_arg in wrong_args:
                if wrong_arg["log_base"] is None:
                    continue
                print(wrong_arg)
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                with pytest.raises(errors):
                    sss.set_value(**wrong_arg)

    def test_set_value_wrong_choice_list_arg(self):
        args = MAKE_GOOD_ARGS["choice"]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            wrong_args = make_arg_choicelist_wrong(arg)
            for wrong_arg in wrong_args:
                if wrong_arg["choice_list"] is None:
                    continue
                print(wrong_arg)
                errors = wrong_arg["error"]
                del wrong_arg["error"]
                with pytest.raises(errors):
                    sss.set_value(**wrong_arg)

    def test_set_value_with_wrong_type(self):
        for type in ALL_TYPE:
            args = MAKE_GOOD_ARGS[type]()
            for arg in args:
                sss = SingleSearchSpace(**arg)
                wrong_args = make_arg_type_wrong(arg)
                for wrong_arg in wrong_args:
                    errors = wrong_arg["error"]
                    del wrong_arg["error"]
                    with pytest.raises(errors):
                        sss.set_value(**wrong_arg)

    def test_align_min_max_to_choice_list_if_categorical(self):
        args = MAKE_GOOD_ARGS["choice"]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            sss._min = -100
            sss._max = -100
            sss._align_min_max_to_choice_list_if_categorical()
            assert sss.min == 0
            assert sss.max == len(arg["choice_list"]) - 1

    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_is_categorical(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            if type in NOT_CATEGORICAL_TYPE:
                assert sss.is_categorical() == False
            else:
                assert sss.is_categorical() == True

    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_use_quantized_step(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            if type in USE_QUANTIZED_STEP_TYPE:
                assert sss.use_quantized_step() == True
            else:
                assert sss.use_quantized_step() == False

    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_use_log_scale(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            if type in USE_LOG_SCALE_TYPE:
                assert sss.use_log_scale() == True
            else:
                assert sss.use_log_scale() == False

    @pytest.mark.parametrize("type", ALL_TYPE)
    def test_lower_space_upper_space(self, type):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            if type == "choice":
                assert sss.lower_space() == 0
                assert sss.upper_space() == len(sss.choice_list) - 1
            else:
                min = arg["min"]
                max = arg["max"]
                if type in USE_LOG_SCALE_TYPE:
                    log_base = arg["log_base"]
                    assert sss.lower_space() == math.log(min, log_base)
                    assert sss.upper_space() == math.log(max, log_base)
                else:
                    assert sss.lower_space() == min
                    assert sss.upper_space() == max

    @pytest.mark.parametrize("type", ALL_TYPE)
    @pytest.mark.parametrize("number", [2.3, 15])
    def test_space_to_real(self, type, number):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            if type == "choice":
                choice_list = arg["choice_list"]
                ret = sss.space_to_real(number)
                expected_ret = min(max(int(number), 0), len(choice_list)-1)
                assert ret == sss.choice_list[expected_ret]
            else:
                ret = sss.space_to_real(number)
                expected_ret = number

                if type in USE_LOG_SCALE_TYPE:
                    log_base = arg["log_base"]
                    expected_ret = log_base ** expected_ret
                if type in USE_QUANTIZED_STEP_TYPE:
                    step = arg["step"]
                    gap = sss.min % step
                    expected_ret = round((expected_ret - gap) / step) * step  + gap
                assert ret == expected_ret

    @pytest.mark.parametrize("type", ALL_TYPE)
    @pytest.mark.parametrize("number", [10, 512.3])
    def test_real_to_space(self, type, number):
        args = MAKE_GOOD_ARGS[type]()
        for arg in args:
            sss = SingleSearchSpace(**arg)
            if type in USE_LOG_SCALE_TYPE:
                log_base = arg["log_base"]
                assert sss.real_to_space(number) == math.log(number, log_base)
            else:
                assert sss.real_to_space(number) == number

class TestSearchSpace:
    @staticmethod
    def get_search_space_depending_on_type(types, range_format=False):
        if not isinstance(types, (list, tuple)):
            types = [types]
        search_space = {}

        if "uniform" in types:
            TestSearchSpace.add_uniform_search_space(search_space, range_format)
        if "quniform" in types:
            TestSearchSpace.add_quniform_search_space(search_space, range_format)
        if "loguniform" in types:
            TestSearchSpace.add_loguniform_search_space(search_space, range_format)
        if "qloguniform" in types:
            TestSearchSpace.add_qloguniform_search_space(search_space, range_format)
        if "choice" in types:
            TestSearchSpace.add_choice_search_space(search_space)

        return search_space

    @staticmethod
    def add_uniform_search_space(search_space, range_format=False):
        search_space["uniform_search_space"] = {"param_type" : "uniform"}
        if range_format:
            search_space["uniform_search_space"]["range"] = [1, 10]
        else:
            search_space["uniform_search_space"].update(
                {"min" : 1, "max" : 10}
            )

    @staticmethod
    def add_quniform_search_space(search_space, range_format=False):
        search_space["quniform_search_space"] = {"param_type" : "quniform"}
        if range_format:
            search_space["quniform_search_space"]["range"] = [1, 10, 3]
        else:
            search_space["quniform_search_space"].update(
                {"min" : 1, "max" : 10, "step" : 3}
            )

    @staticmethod
    def add_loguniform_search_space(search_space, range_format=False):
        search_space["loguniform_search_space"] = {"param_type" : "loguniform"}
        if range_format:
            search_space["loguniform_search_space"]["range"] = [1, 10, 2]
        else:
            search_space["loguniform_search_space"].update(
                {"min" : 1, "max" : 10, "log_base" : 2}
            )

    @staticmethod
    def add_qloguniform_search_space(search_space, range_format=False):
        search_space["qloguniform_search_space"] = {"param_type" : "qloguniform"}
        if range_format:
            search_space["qloguniform_search_space"]["range"] = [1, 10, 3, 2]
        else:
            search_space["qloguniform_search_space"].update(
                {"min" : 1, "max" : 10, "step" : 3, "log_base" : 2}
            )

    @staticmethod
    def add_choice_search_space(search_space):
        search_space["choice_search_space"] = {
            "param_type" : "choice",
            "choice_list" : ["somevalue1", "somevalue2", "somevalue3"]
        }

    @pytest.fixture
    def search_space_with_all_types(self):
        return SearchSpace(self.get_search_space_depending_on_type(ALL_TYPE))

    def test_init_with_range_format_argument(self):
        search_space = self.get_search_space_depending_on_type(ALL_TYPE, True)
        ss = SearchSpace(search_space)
        assert ss is not None

    @pytest.mark.parametrize("type", NOT_CATEGORICAL_TYPE)
    def test_init_with_insufficient_range_arguments(self, type):
        search_space = self.get_search_space_depending_on_type(type, True)
        if type in USE_LOG_SCALE_TYPE:
            num_to_delete = 2
        else:
            num_to_delete = 1
        search_space[f"{type}_search_space"]["range"] = search_space[f"{type}_search_space"]["range"][:-num_to_delete]
        with pytest.raises(ValueError):
            ss = SearchSpace(search_space)

    def test_init_both_format_exists(self):
        search_space = self.get_search_space_depending_on_type(ALL_TYPE)
        range_format = self.get_search_space_depending_on_type(ALL_TYPE, True)
        for key, val in search_space.items():
            val.update(range_format[key])
        ss = SearchSpace(search_space)

    def test_get_item_available(self, search_space_with_all_types):
        for type in ALL_TYPE:
            val = search_space_with_all_types[f"{type}_search_space"]

    def test_iteratble(self, search_space_with_all_types):
        for val in search_space_with_all_types:
            pass

    def test_len_is_available(self, search_space_with_all_types):
        assert len(search_space_with_all_types) == 5

    @pytest.mark.parametrize("choice_exist", [True, False])
    def test_has_categorical_param(self, search_space_with_all_types, choice_exist):
        if choice_exist:
            search_space = self.get_search_space_depending_on_type(ALL_TYPE)
        else:
            search_space = self.get_search_space_depending_on_type(NOT_CATEGORICAL_TYPE)
        ss = SearchSpace(search_space)

        assert ss.has_categorical_param() == choice_exist


    def test_get_real_config_with_proper_argument(self, search_space_with_all_types):
        # search space configuration
        step = 3
        log_base = 2
        min_val = 1
        requested_val = 3.2

        config = {f"{type}_search_space" : requested_val for type in ALL_TYPE}
        real_space = search_space_with_all_types.get_real_config(config)

        for key, val in real_space.items():
            rescaled_requested_val = requested_val
            if key in ["loguniform_search_space", "qloguniform_search_space"]:
                rescaled_requested_val = log_base ** requested_val
            if key in ["quniform_search_space" ,"qloguniform_search_space"]:
                gap = min_val % step
                rescaled_requested_val = round((rescaled_requested_val - gap) / step) * step + gap
            if key == "choice_search_space":
                choice_list = search_space_with_all_types[key].choice_list
                idx = max(min(int(rescaled_requested_val), len(choice_list) - 1), 0)
                rescaled_requested_val = choice_list[idx]

            assert val == rescaled_requested_val

    @pytest.mark.parametrize("wrong_name", ["wrong_name", 1, 3.2])
    def test_get_real_config_with_wrong_name_config(self, search_space_with_all_types, wrong_name):
        config = {wrong_name : 3.2}
        with pytest.raises(KeyError):
            real_space = search_space_with_all_types.get_real_config(config)

    def test_get_space_config_with_proper_argument(self, search_space_with_all_types):
        # search space configuration
        log_base = 2
        requested_val = 10

        config = {f"{type}_search_space" : requested_val for type in ALL_TYPE}
        real_space = search_space_with_all_types.get_space_config(config)

        for key, val in real_space.items():
            rescaled_requested_val = requested_val
            if key in ["loguniform_search_space", "qloguniform_search_space"]:
                rescaled_requested_val = math.log(requested_val, log_base)

            assert val == rescaled_requested_val

    @pytest.mark.parametrize("wrong_name", ["wrong_name", 1, 3.2])
    def test_get_space_config_with_wrong_name_config(self, search_space_with_all_types, wrong_name):
        config = {wrong_name : 3.2}
        with pytest.raises(KeyError):
            real_space = search_space_with_all_types.get_space_config(config)

    def test_get_bayeopt_search_space(self, search_space_with_all_types):
        bayes_opt_format = search_space_with_all_types.get_bayeopt_search_space()

        for val in bayes_opt_format.values():
            assert len(val) == 2
            min_val, max_val = val
            assert min_val < max_val

    def test_convert_from_zero_one_scale_to_real_space_with_good_args(
        self,
        search_space_with_all_types
    ):
        config = {}
        for key in search_space_with_all_types:
            config[key] = 0.5
        real_space = search_space_with_all_types.convert_from_zero_one_scale_to_real_space(config)

    @pytest.mark.parametrize("config", ["wrong_value", [1,3,4], (1,2)])
    def test_convert_from_zero_one_scale_to_real_space_with_bad_arg_type(
        self,
        search_space_with_all_types,
        config
    ):
        with pytest.raises(AttributeError):
            real_space = search_space_with_all_types.convert_from_zero_one_scale_to_real_space(config)

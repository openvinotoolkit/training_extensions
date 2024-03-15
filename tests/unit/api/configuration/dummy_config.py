# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
Dummy configurable parameter class to test configuration functionality
"""

import attr

from otx.api.configuration import (
    ConfigurableParameters,
    ModelLifecycle,
    Operator,
    Rule,
    UIRules,
)
from otx.api.configuration.elements import (
    ConfigurableEnum,
    ParameterGroup,
    add_parameter_group,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    float_selectable,
    selectable,
    string_attribute,
)
from otx.api.configuration.enums import AutoHPOState
from otx.api.configuration.ui_rules import Action


class SomeEnumSelectable(ConfigurableEnum):
    """Test Enum selectable"""

    TEST_NAME1 = "test_name_1"
    TEST_2 = "test_2_test"
    BOGUS_NAME = "bogus"
    OPTION_C = "option_c"


@attr.s
class DatasetManagerConfig(ConfigurableParameters):
    """Dummy configurable parameters class"""

    # type: ignore
    # This class is used for testing purposes only, so mypy should ignore it

    # Component and header are required, description is optional.
    header = string_attribute("Dataset Manager configuration -- TEST ONLY")
    description = string_attribute("Configurable parameters for the DatasetManager -- TEST ONLY")

    # Add some parameters
    number_of_samples_for_auto_train = configurable_integer(
        default_value=5,
        min_value=1,
        max_value=1000,
        header="Samples required for new training round",
    )
    label_constraints = configurable_boolean(default_value=True, header="Apply label constraints")

    @attr.s
    class _NestedParameterGroup(ParameterGroup):
        # A nested group of parameters
        # header is a required attribute, all parameter groups should define one.
        header = string_attribute("Test group of parameter groups")

        @attr.s
        class _SubgroupOne(ParameterGroup):
            # Subgroup one of the nested group, with a couple of parameters
            header = string_attribute("Parameter group one")

            __ui_rules = UIRules(
                rules=[
                    Rule(
                        parameter=["nested_parameter_group", "show_subgroup_one"],
                        operator=Operator.EQUAL_TO,
                        value=False,
                    )
                ],
                action=Action.HIDE,
            )

            bogus_parameter_one = configurable_float(
                default_value=42,
                ui_rules=__ui_rules,
                header="Bogus parameter to test nested parameter groups",
            )
            bogus_parameter_two = configurable_float(
                default_value=42,
                ui_rules=__ui_rules,
                header="Bogus parameter to test nested parameter groups",
            )

        subgroup_one = add_parameter_group(_SubgroupOne)

        show_subgroup_one = configurable_boolean(default_value=True, header="Show the parameters in subgroup one?")

        @attr.s
        class _SubgroupTwo(ParameterGroup):
            # Subgroup two of the nested group, with a couple of parameters
            header = string_attribute("Parameter group two")
            bogus_parameter_three = configurable_float(
                default_value=42,
                header="Bogus parameter to test nested parameter groups",
            )

            bogus_parameter_four = configurable_float(
                default_value=42,
                header="Bogus parameter to test nested parameter groups",
            )

        subgroup_two = add_parameter_group(_SubgroupTwo)

    @attr.s
    class _SubsetParameters(ParameterGroup):
        # Parameters governing sample distribution over subsets
        header = string_attribute("Subset parameters")
        description = string_attribute("Parameters for the different subsets")

        # Add a parameter group 'Subset parameters'
        auto_subset_fractions = configurable_boolean(
            default_value=True,
            description="Test",
            header="Automatically determine subset proportions",
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        __ui_rules = UIRules(
            rules=[
                Rule(
                    parameter="auto_subset_fractions",
                    value=False,
                    operator=Operator.EQUAL_TO,
                )
            ],
            action=Action.SHOW,
        )

        train_proportion = configurable_float(
            default_value=0.75,
            min_value=0.0,
            max_value=1.0,
            header="Training set proportion",
            ui_rules=__ui_rules,
            affects_outcome_of=ModelLifecycle.TRAINING,
            auto_hpo_state=AutoHPOState.POSSIBLE,
        )

        validation_proportion = configurable_float(
            default_value=0.1,
            min_value=0.0,
            max_value=1.0,
            header="Validation set proportion",
            ui_rules=__ui_rules,
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

        test_proportion = configurable_float(
            default_value=0.15,
            min_value=0.0,
            max_value=1.0,
            header="Test set proportion",
            ui_rules=__ui_rules,
            affects_outcome_of=ModelLifecycle.TRAINING,
        )

    # Add a selectable and float selectable parameter
    dummy_float_selectable = float_selectable(
        options=[1.0, 2.0, 3.0, 4.0],
        default_value=2.0,
        header="Test float selectable",
        auto_hpo_state=AutoHPOState.POSSIBLE,
    )

    dummy_selectable = selectable(
        default_value=SomeEnumSelectable.BOGUS_NAME,
        header="Test",
        affects_outcome_of=ModelLifecycle.INFERENCE,
    )

    # Finally, add the nested parameter group and subset parameter groups to the config
    # NOTE! group initialization should use a factory to avoid passing mutable default arguments. This is why the
    # add_parameter_group function is needed.
    nested_parameter_group = add_parameter_group(_NestedParameterGroup)
    subset_parameters = add_parameter_group(_SubsetParameters)

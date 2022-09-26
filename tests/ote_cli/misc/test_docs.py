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

import pytest
import os
from collections import defaultdict
from subprocess import run  # nosec

from ote_cli.registry import Registry

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

class TestDocs:
    @e2e_pytest_component
    def test_help_stdoutputs_of_tools(self):
        with open("QUICK_START_GUIDE.md", encoding="UTF-8") as read_file:
            commands = []
            full_text = ''
            for line in read_file:
                full_text += line
                if "ote" in line and "--help" in line:
                    commands.append(line.strip().split(' '))

            MAX_INDENT = 10

            for command in commands:
                output = run(command, capture_output=True)
                help_message = output.stdout.decode()
                found = True
                if help_message not in full_text:
                    found = False
                    for _ in range(MAX_INDENT):
                        help_message = "\n".join([" " + line for line in help_message.split("\n")])
                        if help_message in full_text:
                            found = True
                            break
                assert found, f"\nHelp message:\n{output.stdout.decode()}\n was not found in \n{full_text}"

    @e2e_pytest_component
    def test_algorithms_table(self):
        def algorithms_table(templates):
            attributes = ["model_template_id", "name", "gigaflops", "size"]
            algo_table = defaultdict(list)

            for template in sorted(templates, key=lambda x: float(x.gigaflops)):
                if template.model_template_path.split("/")[-1] != "template.yaml":
                    continue
                record = [str(getattr(template, attr)) for attr in attributes]
                record.append(os.path.relpath(template.model_template_path, './external'))
                algo_table[record[0]] = record[1:]
            return algo_table

        readme_table = defaultdict(list) # ["name", "gigaflops", "size", "Path"]
        with open("external/README.md", encoding="UTF-8") as read_file:
            full_text = ''
            for line in read_file:
                full_text += line
                if "|" in line:
                    model_lst = line.replace(" ", "").strip().split("|")
                    model_lst = " ".join(model_lst).split()
                    readme_table[model_lst[0]] = model_lst[1:]

        registry = Registry(".")
        templates_per_task_type = defaultdict(list)
        for template in sorted(registry.templates, key=lambda x:str(x.task_type)):
            templates_per_task_type[template.task_type].append(template)
        for task_type, templates in templates_per_task_type.items():
            algorithm_table = algorithms_table(templates)
            for model_id in algorithm_table.keys():
                assert model_id in readme_table, f"\n {model_id} not in 'external/README.md' for {task_type}"
                assert algorithm_table[model_id] == readme_table[model_id], f"\n {model_id}'s info in 'external/README.md' is wrong"

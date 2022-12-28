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

import os
from collections import defaultdict
from subprocess import run  # nosec

from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component


class TestDocs:
    @e2e_pytest_component
    def test_help_stdoutputs_of_tools(self):
        # read and gathering help command example and corresponding output from the doc
        with open("QUICK_START_GUIDE.md", encoding="UTF-8") as read_file:
            commands = []
            msg_in_doc = []
            msg_temp = ""
            full_text = ''
            found_help_cmd = False
            cmd_idx = 0
            for line in read_file:
                full_text += line
                if "$" in line and "ote" in line and "--help" in line:
                    commands.append(line.split("$")[-1].strip().split(' '))
                    found_help_cmd = True
                elif found_help_cmd:
                    # grep all messages quoted with "```"
                    if "```" not in line:
                        if line == "...\n":
                            # some msg output is too long to put them all onto the doc.
                            # those will be marked as "..." to represent its shrinking
                            continue
                        msg_temp += msg_temp
                    else:
                        # tokenize msg in the doc by replacing whitespace to a single space
                        # for ease comparison with actual output messages
                        msg_in_doc.append(" ".join(msg_temp.split()))
                        msg_temp = ""
                        cmd_idx += 1
                        found_help_cmd = False

            assert len(commands) == len(msg_in_doc), \
                f"length of cmds & msg in doc is mismatched. len(cmds) = {len(commands)}, " \
                f"len(msg_in_doc) = {len(msg_in_doc)}"

            # compare actual help message output & one that came from doc
            for idx in range(len(commands)):
                output = run(commands[idx], capture_output=True)
                help_msg = output.stdout.decode()
                # tokenize by replace all whitespace to a single space
                help_msg = " ".join(help_msg.split())
                # asserting with "in" op to deal with shrinked message as well
                assert msg_in_doc[idx] in help_msg, \
                    f"help message in doc:\n{msg_in_doc[idx]}\nis not equal with stdout:\n{help_msg}"

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

        readme_table = defaultdict(list)  # ["name", "gigaflops", "size", "Path"]
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
        for template in sorted(registry.templates, key=lambda x: str(x.task_type)):
            templates_per_task_type[template.task_type].append(template)
        for task_type, templates in templates_per_task_type.items():
            algorithm_table = algorithms_table(templates)
            for model_id in algorithm_table.keys():
                assert model_id in readme_table, \
                    f"\n {model_id} not in 'external/README.md' for {task_type}"
                assert algorithm_table[model_id] == readme_table[model_id], \
                    f"\n {model_id}'s info in 'external/README.md' is wrong"

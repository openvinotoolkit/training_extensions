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

from ote_cli.registry import Registry


def test_help_stdoutputs_of_tools():
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


def test_algorithms_table():
    def algorithms_generate_table(templates):
        attributes = ["model_template_id", "name", "gigaflops", "size"]
        header = attributes + ["Path"]
        attributes_in_md = {"name": "Name", "model_template_id": "ID", "gigaflops": "Complexity (GFlops)", "size": "Model size (MB)", "Path": "Path"}
        
        table = [" | ".join([attributes_in_md[x] for x in header])] + [" | ".join(["-------" for _ in header])]
        
        for template in sorted(templates, key=lambda x: float(x.gigaflops)):
            record = [str(getattr(template, attr)) for attr in attributes ]
            record.append(os.path.relpath(template.model_template_path, './external'))
            record = " | ".join(record)
            table += [record]
        return "\n".join(table)
    
    with open("external/README.md", encoding="UTF-8") as read_file:
        full_text = ''
        for line in read_file:
            full_text += line
    
    registry = Registry(".")
    templates_per_task_type = defaultdict(list)
    for template in sorted(registry.templates, key=lambda x:str(x.task_type)):
        templates_per_task_type[template.task_type].append(template)
    for task_type, templates in templates_per_task_type.items():
        generated_table = algorithms_generate_table(templates)
        print("\n", task_type)
        print(generated_table)
        assert generated_table in full_text, f"\n{generated_table} not in \n{full_text}\n for the task {task_type}\n"

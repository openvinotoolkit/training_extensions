"""
 Copyright (c) 2018 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from argparse import ArgumentParser
import yaml

class ArgumentParserWithYaml(ArgumentParser):
    """
    Attention, this will work with simple yaml files only, and if there is no action=store_false
    """
    @staticmethod
    def _check_arg_line_repr_None(arg_line, k, v):
        """ The method is required, since by default python prints None value as None, whereas yaml waiths for null """
        s = arg_line.strip()
        prefixes = [k, "'" + k + "'", '"' + k + '"']
        is_ok = False
        for prefix in prefixes:
            if s.startswith(prefix):
                s = s[len(prefix):]
                is_ok = True
                break
        if not is_ok:
            raise RuntimeError("Unknown prefix in line '{}', k = '{}', v = '{}'".format(arg_line, k, v))
        s = s.strip()
        assert s.startswith(':'), "Bad format of line '{}', k = '{}', v = '{}'".format(arg_line, k, v)
        s = s[1:]
        s = s.strip()
        #print("arg line '{}' repr None = {}, s = '{}'".format(arg_line, s == "None", s))

        return s == "None" #note that 'None' will be a string, whereas just None will be None

    def convert_arg_line_to_args(self, arg_line):
        arg_line = arg_line.strip()
        if not arg_line:
            return []
        if arg_line.endswith(','):
            arg_line = arg_line[:-1]

        data = yaml.load(arg_line)
        if data is None:
            return []
        assert type(data) is dict
        assert len(data) == 1

        res = []
        for k, v in data.items():
            if v == 'None': # default value is None -- skipping
                if self._check_arg_line_repr_None(arg_line, k, v): #additional check that somebody passed string "None"
                    continue
                else:
                    print("WARNING: DURING PARSING ARGUMENTS FILE: possible error in the argument line '{}' -- probably None value is missed".format(arg_line))

            if type(v) is list:
                res.append('--' + str(k))
                [res.append(str(item)) for item in v]
                continue

            if type(v) is bool: # special case, action=store_true, do not use store_false!
                if v:
                    res.append('--' + str(k))
                continue

            # attention, there may be small issue with converting float -> string -> float -> string
            res.extend(['--' + str(k), str(v)])

        return res

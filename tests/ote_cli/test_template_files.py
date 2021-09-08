import os

from ote_sdk.entities.model_template import parse_model_template

from ote_cli.registry import Registry

def gen_parse_model_template_tests():
    class MyTests:
        pass

    templates_dir = 'external'
    registry = Registry(templates_dir)
    for template in registry.templates:
        path = os.path.relpath(template['path'])
        def test_template(self, path=path):
            template = parse_model_template(path)
            print(template.task_type)
            assert template.hyper_parameters.data
        setattr(MyTests, 'test_' + path.replace(' ', '_').replace('/', '_'), test_template)

    return MyTests


class TestParseModelTemplates(gen_parse_model_template_tests()):
    pass


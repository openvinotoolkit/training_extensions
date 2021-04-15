import json
from pathlib import Path

model_list = []
TEST_ROOT = Path(__file__).parent
eval_config = json.load(open(TEST_ROOT / 'ote_accuracy_validation.json'))
for domain_name in eval_config:
    model_type = eval_config[domain_name]
    for problem_name in model_type:
        model_dict = model_type[problem_name]
        for model in model_dict:
            model_list.append(model)

with open('model_list.txt', 'w') as f:
    for mn in model_list:
        f.write(f'{mn}\n')

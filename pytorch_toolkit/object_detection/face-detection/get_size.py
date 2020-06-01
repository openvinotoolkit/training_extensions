import os
import yaml

def get_sha256(path, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    os.system(f'sha256sum {path} > {work_dir}/sha256.txt')
    with open(f'{work_dir}/sha256.txt') as f:
        sha256 = f.readlines()[0].strip().split(' ')[0]
    return sha256

def get_size(path, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    os.system(f'ls -l {name} > {work_dir}/ls.txt')
    with open(f'{work_dir}/ls.txt') as f:
        size = f.readlines()[0].strip().split(' ')[4]
    return int(size)

for root, dirs, files in os.walk(".", topdown=False):
   for name in files:
       if name.endswith('.yml'):
           model_yml = os.path.join(root, name)
           with open(model_yml) as f:
               content = yaml.load(f)
           source = content['files'][0]['source']
           name = content['files'][0]['name']
           #os.system(f'wget {source}')
           content['files'][0]['sha256'] = get_sha256(name, '/tmp/1')
           content['files'][0]['size'] = get_size(name, '/tmp/1')
           with open(model_yml, 'w') as f:
               yaml.dump(content, f)

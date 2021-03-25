import subprocess
from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, listdir, walk, popen
from os.path import exists, join, isfile, abspath, basename
from shutil import rmtree

from joblib import delayed, Parallel


VIDEO_EXTENSIONS = 'avi', 'mp4', 'mov', 'webm'


def create_dirs(dir_path, override=False):
    if override:
        if exists(dir_path):
            rmtree(dir_path)
        makedirs(dir_path)
    elif not exists(dir_path):
        makedirs(dir_path)


def parse_relative_paths(data_dir, extensions):
    data_dir = abspath(data_dir)
    skip_size = len(data_dir) + 1

    relative_paths = []
    for root, sub_dirs, files in walk(data_dir):
        if len(sub_dirs) == 0 and len(files) > 0:
            valid_files = [f for f in files if f.split('.')[-1].lower() in extensions]
            if len(valid_files) > 0:
                relative_paths.append(root[skip_size:])

    return relative_paths


def prepare_tasks(relative_paths, input_dir, output_dir, extensions):
    out_tasks = []
    for relative_path in relative_paths:
        input_videos_dir = join(input_dir, relative_path)
        assert exists(input_videos_dir)

        input_video_files = [f for f in listdir(input_videos_dir)
                             if isfile(join(input_videos_dir, f)) and f.split('.')[-1].lower() in extensions]
        if len(input_video_files) == 0:
            continue

        for input_video_file in input_video_files:
            input_video_path = join(input_videos_dir, input_video_file)

            video_name = input_video_file.split('.')[0].lower()
            output_video_dir = join(output_dir, relative_path, video_name)

            if exists(output_video_dir):
                existed_files = [f for f in listdir(output_video_dir) if isfile(join(output_video_dir, f))]
                existed_frame_ids = [int(f.split('.')[0]) for f in existed_files]
                existed_num_frames = len(existed_frame_ids)
                if min(existed_frame_ids) != 1 or max(existed_frame_ids) != existed_num_frames:
                    rmtree(output_video_dir)
                else:
                    continue

            out_tasks.append((input_video_path, output_video_dir, join(relative_path, video_name)))

    return out_tasks


def dump_frames(video_path, out_dir, image_name_template, max_image_size):
    result = popen(
        f'ffprobe -hide_banner '
        f'-loglevel error '
        f'-select_streams v:0 '
        f'-show_entries stream=width,height,r_frame_rate '
        f'-of csv=p=0 {video_path}'
    )
    video_width, video_height, video_fps = result.readline().rstrip().split(',')

    video_width = int(video_width)
    video_height = int(video_height)

    video_fps_parts = video_fps.split('/')
    assert len(video_fps_parts) == 2
    video_fps = float(video_fps_parts[0]) / float(video_fps_parts[1])

    if video_width > video_height:
        command = ['ffmpeg',
                   '-hide_banner',
                   '-threads', '1',
                   '-loglevel', '"panic"',
                   '-i', '"{}"'.format(video_path),
                   '-vsync', '0',
                   '-vf', '"scale={}:-2"'.format(max_image_size),
                   '-q:v', '5',
                   '"{}"'.format(join(out_dir, image_name_template)),
                   '-y']
    else:
        command = ['ffmpeg',
                   '-hide_banner',
                   '-threads', '1',
                   '-loglevel', '"panic"',
                   '-i', '"{}"'.format(video_path),
                   '-vsync', '0',
                   '-vf', '"scale=-2:{}"'.format(max_image_size),
                   '-q:v', '5',
                   '"{}"'.format(join(out_dir, image_name_template)),
                   '-y']
    command = ' '.join(command)

    try:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return None

    num_dumped_frames = len([True for f in listdir(out_dir) if isfile(join(out_dir, f))])
    if num_dumped_frames == 0:
        return None

    return num_dumped_frames, video_fps


def process_task(input_video_path, output_video_dir, relative_path, image_name_template, max_image_size):
    create_dirs(output_video_dir)

    out_info = dump_frames(
        input_video_path, output_video_dir, image_name_template, max_image_size
    )
    if out_info is None:
        rmtree(output_video_dir)
        return None

    video_num_frames, video_fps = out_info

    return relative_path, video_num_frames, video_fps


def load_annotation(input_files):
    out_data = dict()
    for input_file in input_files:
        assert exists(input_file), f'File "{input_file}" does not exist'

        annot_file_name = basename(input_file).split('.')[0]
        with open(input_file) as input_stream:
            for line in input_stream:
                line_parts = line.split(' ')
                if len(line_parts) != 2:
                    continue

                video_rel_path, class_id = line_parts
                class_id = int(class_id)

                annot_key = video_rel_path.split('.')[0].lower()
                out_data[annot_key] = class_id, annot_file_name

    return out_data


def merge_annot(records, annot):
    out_data = []
    for record in records:
        if record is None:
            continue

        rel_path, num_frames, fps = record
        if num_frames <= 0:
            continue

        annot_key = rel_path.lower()
        if annot_key not in annot:
            continue

        class_id, annot_file_name = annot[annot_key]

        out_data.append(dict(
            rel_path=rel_path,
            num_frames=num_frames,
            fps=fps,
            class_id=class_id,
            annot_file_name=annot_file_name
        ))

    return out_data


def split_by_annot_type(records):
    out_data = defaultdict(list)
    for record in records:
        out_data[record['annot_file_name']].append(record)

    return out_data


def dump_records(all_records, out_dir):
    for ann_file, records in all_records.items():
        with open(join(out_dir, ann_file), 'w') as output_stream:
            for record in records:
                rel_path = record['rel_path']
                label = record['class_id']
                num_frames = record['num_frames']
                fps = record['fps']

                converted_record = rel_path, label, 0, num_frames, 0, num_frames, fps
                output_stream.write(' '.join([str(r) for r in converted_record]) + '\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--videos_dir', '-i', type=str, required=True)
    parser.add_argument('--annotation', '-a', type=str, nargs='+', required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument('--out_extension', '-ie', type=str, required=False, default='jpg')
    parser.add_argument('--max_image_size', '-ms', type=int, required=False, default=256)
    parser.add_argument('--override', action='store_true', required=False)
    parser.add_argument('--num_jobs', '-n', type=int, required=False, default=12)
    args = parser.parse_args()

    assert exists(args.videos_dir)
    assert len(args.annotation) > 0

    out_frames_dir = join(args.output_dir, 'rawframes')
    create_dirs(out_frames_dir, override=args.override)

    print('\nLoading annotation ...')
    annot = load_annotation(args.annotation)
    print(f'Loaded {len(annot)} records.')

    print('\nPreparing tasks ...')
    relative_paths = parse_relative_paths(args.videos_dir, VIDEO_EXTENSIONS)
    tasks = prepare_tasks(relative_paths, args.videos_dir, out_frames_dir, VIDEO_EXTENSIONS)
    print(f'Prepared {len(tasks)} tasks.')

    print('\nDumping frames ...')
    image_name_template = f'%05d.{args.out_extension}'
    records = Parallel(n_jobs=args.num_jobs, verbose=True)(
        delayed(process_task)(*task,
                              image_name_template=image_name_template,
                              max_image_size=args.max_image_size)
        for task in tasks
    )
    print('Finished.')

    print('\nMerging annotation info ...')
    records = merge_annot(records, annot)
    print(f'Merged {len(records)} records.')

    print('\nDumping annotation ...')
    records_by_annot_type = split_by_annot_type(records)
    dump_records(records_by_annot_type, args.output_dir)
    print('Finished.')


if __name__ == '__main__':
    main()

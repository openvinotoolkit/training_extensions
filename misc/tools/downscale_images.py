"""
  Script for downscaling images. Can be used for improving training speed.
"""

import argparse
import os

import cv2


def parse_args():
  """
    Parse command arguments.
  """
  parser = argparse.ArgumentParser(description='Downscale images inplace')
  parser.add_argument('path', help='Path to a directory with images')
  parser.add_argument('-target_size', type=int, default=450,
                      help='A minimum size of a image side (weight or height)')
  return parser.parse_args()


def downscale(paths: list, min_size: int, save_aspect_ratio: bool):
  """ Downscale images inplace

  Args:
    paths: Full paths to images
    min_size: A minimum size of a image side (weight or height)
    save_aspect_ratio: Save aspect ratio of an origin image
  """

  def _resize_and_save(path, img, target_width, target_height):
    img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img)

  downscaled_count = 0
  skipped_count = 0
  smaller_count = 0
  for id_, key in enumerate(paths):
    path = key
    img = cv2.imread(path)
    if img is None:
      print('Skip: {}'.format(path))
      continue
    height, width = img.shape[:2]
    if save_aspect_ratio:
      if min_size in (width, height):
        print("{0:04d}: {1}    ({2}) == ({3}, {4})".format(id_, path, min_size, width, height))
        skipped_count += 1
      elif height > min_size and width > min_size:
        target_width = int(width * min_size / height)
        target_height = int(height * min_size / width)

        if target_height >= target_width:
          target_width = min_size
        else:
          target_height = min_size

        _resize_and_save(path, img, target_width, target_height)
        print("{0:04d}: {1}    ({2}, {3}) -> ({4}, {5})".format(id_, path, width, height, target_width, target_height))
        downscaled_count += 1
      else:
        print("{0:04d}: {1}    ({2}) < max({3}, {4})".format(id_, path, min_size, width, height))
        smaller_count += 1
    else:
      _resize_and_save(path, img, min_size, min_size)
      print("{0:04d}: {1}    ({2}, {3}) -> ({4}, {5})".format(id_, path, width, height, min_size, min_size))
      downscaled_count += 1

  print('Summary (min_size = {0}, save_aspect_ratio = {1}):'.format(min_size, save_aspect_ratio))
  print('  downscaled images     = {0}'.format(downscaled_count))
  print('  equal min_size images = {0}'.format(skipped_count))
  print('  small images          = {0}'.format(smaller_count))


def main():
  args = parse_args()
  root_dir = args.path

  for (dirpath, _, filenames) in os.walk(root_dir):
    print('Process {0}'.format(dirpath))
    filenames = sorted([os.path.join(dirpath, file) for file in filenames])
    downscale(filenames, min_size=args.target_size, save_aspect_ratio=True)


if __name__ == "__main__":
  main()

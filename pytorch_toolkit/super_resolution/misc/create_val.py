import argparse
import os
import os.path as osp
from matplotlib import pyplot as plt
import skimage
from skimage import io
from skimage import transform
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Val dataset generator. It crops and resizes given images to specified size.")

parser.add_argument("--input_folder_path", default="", type=str, help="Path to folder with input images", required=True)
parser.add_argument("--output_folder_path", default="", type=str, help="Path to output folder", required=True)
parser.add_argument("--count", default=None, type=int, help="Number of cropped images")
parser.add_argument("--w", default=None, type=int, help="Width of cropped image")
parser.add_argument("--h", default=None, type=int, help="Height of cropped image")
parser.add_argument("--ds_factor", default=4, type=int, help="Downscaling factor")
parser.add_argument("--show", action='store_true', help="Show results")

def main():
    opt = parser.parse_args()

    input_folder = opt.input_folder_path
    output_folder = opt.output_folder_path
    target_h = opt.h
    target_w = opt.w
    count = opt.count
    ds_factor = opt.ds_factor

    images_names = os.listdir(input_folder)
    images_names.sort()

    if count is None:
        count = len(images_names)

    for i in tqdm(range(count)):
        name = images_names[i]
        path = osp.join(input_folder, name)

        image = io.imread(path)
        h = image.shape[0]
        w = image.shape[1]

        if target_h is None:
            target_h = h

        if target_w is None:
            target_w = w

        if h >= target_h and w >= target_w:
            dw = (w - target_w) // 2
            dh = (h - target_h) // 2

            cropped = image[dh:target_h+dh, dw:target_w+dw]
            cropped_h = cropped.shape[0]
            cropped_w = cropped.shape[1]

            assert(cropped_h % ds_factor == 0)
            assert(cropped_w % ds_factor == 0)

            im = skimage.img_as_float32(cropped)
            resized = transform.resize(image=im, output_shape=(cropped_h / ds_factor, cropped_w / ds_factor), order=3,
                                          mode='reflect', anti_aliasing=True,
                                          anti_aliasing_sigma=None, preserve_range=True)


            name_wo_ext = os.path.splitext(name)[0]
            out_resized_image_path = osp.join(output_folder, name_wo_ext + "_lr_x" + str(ds_factor) + ".png")
            io.imsave(out_resized_image_path, resized)

            out_image_path = osp.join(output_folder, name_wo_ext + "_hr.png")
            io.imsave(out_image_path, cropped)

            if opt.show:
                f1 = plt.figure()
                plt.imshow(cropped)
                f2 = plt.figure()
                plt.imshow(resized)
                plt.show()

if __name__ == "__main__":
    main()
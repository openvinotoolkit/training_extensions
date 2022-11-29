import os
import re
import numpy as np
from medpy import io
from PIL import Image
from skimage.util import view_as_windows

# Defining criteria to extract patches
PATCH_UP_BOUND = 0.7
PATCH_LOW_BOUND = 0.3

def consider_patch(patch):
    counts, _ = np.histogram(patch * 255., bins=256)
    if counts[0] < sum(counts)/3 and \
            counts[-1] < sum(counts)/3 \
            and patch.mean() > PATCH_LOW_BOUND and patch.mean() < PATCH_UP_BOUND:
        return True
    else:
        return False

def main(args):
    # counters for gathering statistics
    train_count, test_count = 0, 0
    unint16_count, uint8_count = 0, 0
    all_patch_count, valid_patch_count = 0, 0

    try:
        for root, folders, files in os.walk(os.path.abspath(args.path_to_dataset)):

            # only folders containing just one .dcm (000000.dcm) are of interest
            if len(files) == 1:
                dcm_path = os.path.join(root, files[0])
                print(dcm_path)
                # get the patient ID and training indicator
                patient_id = re.findall(r"\D(P_\d{5})\D", dcm_path)[
                    0]  # search for 'P_?????' pattern
                is_training = 'Training' in dcm_path

                image, _ = io.load(dcm_path)  # medpy's .dcm loader
                image = np.squeeze(image)  # squeeze out '1' length dimensions

                # only 16bit mammograms are considered
                if image.dtype == 'uint16' and \
                        image.shape[0] > args.patch_dim and image.shape[1] > args.patch_dim:
                    # image should be atleast as big as the window (patch_dim x patch_dim)

                    unint16_count += 1
                    image = image.astype(np.float32)/(2**16 - 1)  # normalize b/w [0,1]

                    if is_training:
                        # Extract (non-overlapping) patches for training
                        patches = view_as_windows(image, window_shape=(
                            args.patch_dim, args.patch_dim), step=args.patch_dim)
                        for rowid, row in enumerate(patches):
                            for colid, patch in enumerate(row):
                                all_patch_count += 1
                                if consider_patch(patch):
                                    patch = Image.fromarray(
                                        patch * 255.).convert('L')
                                    save_path = os.path.join(os.path.abspath(args.out_train),
                                                             '_'.join([str(patient_id), str(rowid), str(colid)]) + '.png')
                                    patch.save(save_path)
                                    valid_patch_count += 1
                        train_count += 1
                    else:
                        # full mammograms for testing

                        # round to 'nearest integer divisible by 32'
                        exact_shape = (round(image.shape[0]/32.)-1) * 32, (round(image.shape[1]/32.)-1) * 32

                        # make the full mammograms shape a multiple of 32
                        image = Image.fromarray(image[0:exact_shape[0], 0:exact_shape[1]] * 255.).convert('L')
                        save_path = os.path.join(os.path.abspath(args.out_test),
                                                 str(patient_id) + '.png')
                        image.save(save_path)
                        test_count += 1
                else:
                    # skip these
                    uint8_count += 1
    except KeyboardInterrupt:
        print('Terminating patch generation')
    finally:
        if args.report:
            print('Total mammograms: {}'.format(unint16_count + uint8_count))
            print('16Bit mammograms: {}'.format(unint16_count))
            print('8Bit mammograms: {} (ignored)'.format(unint16_count))
            print('Training mammograms: {}'.format(train_count))
            print('Training mammograms: {}'.format(test_count))
            print('Total {} valid patches out of {} patches'.format(valid_patch_count, all_patch_count))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
	"""
	Data preperation script. It may take a little while to generate all patches and full-scale images. 
	You can interrupt the generation in the middle by CTRL+C.

	Provide the folder path of original dataset.
	This folder has to contain folders named 'Calc-Test_P_00127_RIGHT_MLO' etc.
	Provide two directories (--out_train and --out_test) to store trainig images (patches) and testing images (full-scale mammograms).

	(Optionally) Provide the patch size (--patch_dim). We strongly recommend the default values (256 or 128).
	(Optionally) You may choose to generate a statistical report.
	"""
    )

    parser.add_argument('-p', '--path_to_dataset', type=str, required=True, help='Path to the original CBIS-DDSM folder')
    parser.add_argument('-d', '--patch_dim', type=int, required=False, default=256, help='Patches of size (patch_dim x patch_dim)')
    parser.add_argument('--out_train', type=str, required=True, help='Output folder for training samples')
    parser.add_argument('--out_test', type=str, required=True, help='Output folder for testing samples')
    parser.add_argument('--report', action='store_true', help='Report statistics')

    args = parser.parse_args()
    main(args)

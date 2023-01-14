import SimpleITK as sitk
import pylidc as pl
import numpy as np
from collections import defaultdict
import os
import cv2
from tqdm import tqdm as tq


def make_mask(height, width, slice_list, *args, **kwargs):
    """Creates masks from the annotations given.

    Parameters
    ----------
    height: int
        Height of the mask to be created
    widht: int
        Width of mask image to be created
    slice_list: list

    ii,jj: dict
        Dictionary containing annotations

    Returns
    -------
    nparray


    """
    mask = np.zeros((height, width))
    n = kwargs.get('n', None)
    point_dictx = kwargs.get('ii', None)
    point_dicty = kwargs.get('jj', None)

    if n in slice_list:
        temp_listx = point_dictx[n]
        temp_listy = point_dicty[n]
        plot_listx = [sum(x)/len(point_dictx[n]) for x in zip(*temp_listx)]
        plot_listy = [sum(y)/len(point_dicty[n]) for y in zip(*temp_listy)]
        merged_list = np.array([[plot_listy[i], plot_listx[i]]
                               for i in range(0, len(plot_listx))])

        cv2.fillPoly(mask, pts=np.int32([merged_list]), color=(255, 255, 255))

    return mask


def extract_slices(dataset_path, save_path, masktype='nodule'):
    """Extracts induvidual slices from the CT volumes given
    in the dataset, clips the max-min values and stores them
    as numpy arrays.

    Parameters
    ----------
    dataset_path: str
        Folder location of the dataset
    save_path: str
        Folder location to save the induvidual image & masks.
    masktype: str
        Nodule mask or Lung mask

    Returns
    -------

    None
    """

    file_list = []

    for tr in tq(range(10)):
        subset_path = dataset_path+"/subset"+str(tr)+"/"
        for file in os.listdir(subset_path):
            if file.endswith(".mhd"):
                file_list.append(os.path.join(subset_path, file))

    for file in tq(file_list):
        file_name = os.path.basename(file)
        series_instance_uid = os.path.splitext(file_name)[0]
        img_file = file

        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        num_slice, height, width = img_array.shape
        # Has the image data

        scan = pl.query(pl.Scan).filter(
            pl.Scan.series_instance_uid == series_instance_uid).first()

        # Maped the image data with annotation using series id

        # Function used to determine which annotation belongs to which nodule
        nods = scan.cluster_annotations()

        nodule_dict = {}  # Dict to store number of contour markings for that nodule
        slice_list = []  # List to store the slices which has nodules marked
        # These dicts are to store the points to be plotted (key=slice_index, )
        points_dictx = {}
        points_dicty = {}
        points_dictx = defaultdict(lambda: [], points_dictx)
        points_dicty = defaultdict(lambda: [], points_dicty)
        for nod in nods:
            # Stores a dict which has count of annotation for each nodule
            nodule_dict[i] = len(nod)

        for key, value in nodule_dict.items():
            # if value>=3 :    #Taking annotations provided by 3 or more annotator
            for i in range(value):
                ann = nods[key][i]  # -1 to keep index correct
                # All coutours for specific nodule collected
                con = ann.contours[0]

                k = con.image_k_position  # Returns the slice number/index which has the nodule
                slice_list.append(k)
                ii, jj = ann.contours[0].to_matrix(include_k=False).T
                points_dictx[k].append(ii)
                points_dicty[k].append(jj)

        # !!Note!! The pylidc package gives cordinates for single slices, If more than one annotaions are give then
        # Sum(x)/total no: of annotation for all provided pixel is given as input

        for n in range(1, num_slice):

            image = (img_array[n].copy()).astype(np.float32)
            im_max = np.max(image)
            if im_max != 0:
                image[image > 1000] = 1000
                image[image < -1000] = -1000
                mask = make_mask(height, width, slice_list,
                                 ii=points_dictx, jj=points_dicty, n=n)
                mask = np.array(mask, dtype=np.float32)
                image = image - image.min()
                image = image/image.max()

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)

                if not os.path.isdir(save_path+'/img'):
                    os.makedirs(save_path+'/img')
                    np.save(save_path+'/img/'+series_instance_uid +
                            '_slice'+str(n)+'.npy', image)
                else:
                    np.save(save_path+'/img/'+series_instance_uid +
                            '_slice'+str(n)+'.npy', image)

                if not os.path.isdir(save_path+'/mask'):
                    os.makedirs(save_path+'/mask')
                    np.save(save_path+'/mask/'+series_instance_uid +
                            '_slice'+str(n)+'.npy', mask)
                else:
                    np.save(save_path+'/mask/'+series_instance_uid +
                            '_slice'+str(n)+'.npy', mask)


def generate_lungseg(dataset_path, save_path):
    file_list = []

    for file in os.listdir(dataset_path):
        if file.endswith(".mhd"):
            file_list.append(os.path.join(dataset_path, file))

    for img_file in tq(file_list):
        file_name = os.path.basename(img_file)
        series_instance_uid = os.path.splitext(file_name)[0]
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)
        num_slice, _, _ = img_array.shape
        for n in range(1, num_slice):
            if not os.path.isdir(save_path+'/lungseg'):
                os.makedirs(save_path+'/lungseg')
                np.save(save_path+'/lungseg/'+series_instance_uid +
                        '_slice'+str(n)+'.npy', img_array[n])
            else:
                np.save(save_path+'/lungseg/'+series_instance_uid +
                        '_slice'+str(n)+'.npy', img_array[n])

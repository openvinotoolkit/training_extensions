import numpy as np
import csv


def read_data(file_name, has_visibility):
    all_keypoints_coords = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            keypoint_coords = row[1:]
            for i in range(len(keypoint_coords)):
                if keypoint_coords[i] == 'nan':
                    keypoint_coords[i] = '-1'
                keypoint_coords[i] = float(keypoint_coords[i])
            all_keypoints_coords.append(keypoint_coords)

    data = np.array(all_keypoints_coords)
    num_dims = 2
    if has_visibility:
        num_dims = 3
    data = np.reshape(data, [data.shape[0], int(data.shape[1] / num_dims), num_dims])

    vis_label = np.zeros((data.shape[0], data.shape[1]))

    if has_visibility:
        vis_label[:, :] = data[:, :, 2]
        data = data[:, :, 0:2]
    else:
        vis_label = vis_label + 1
        data[data < 0] = 1

    return data, vis_label


def get_head_size(gt):
    head_size = np.linalg.norm(gt[:, 9, :] - gt[:, 8, :], axis=1)
    for n in range(gt.shape[0]):
        if gt[n, 8, 0] < 0 or gt[n, 9, 0] < 0:
            head_size[n] = 0

    return head_size


def get_normalized_distance(gt, prediction, head_size):
    num_images = prediction.shape[0]
    num_keypoints = prediction.shape[1]
    distances = np.zeros([num_images, num_keypoints])
    for img_id in range(num_images):
        current_head_size = head_size[img_id]
        if current_head_size == 0:
            distances[img_id, :] = -1
        else:
            distances[img_id, :] = np.linalg.norm(gt[img_id, :, :] - prediction[img_id, :, :], axis=1) / current_head_size
            for kpt_id in range(num_keypoints):
                if gt[img_id, kpt_id, 0] < 0 or gt[img_id, kpt_id, 1] < 0:
                    distances[img_id, kpt_id] = -1
    return distances


def compute_pckh(distances, pckh_threshold_range):
    num_keypoints = distances.shape[1]
    pckh = np.zeros([len(pckh_threshold_range), num_keypoints + 2])

    for kpt_id in range(num_keypoints):
        for threshold_id in range(len(pckh_threshold_range)):
            threshold = pckh_threshold_range[threshold_id]
            joint_distance = distances[:, kpt_id]
            pckh[threshold_id, kpt_id] = np.mean(joint_distance[np.where(joint_distance >= 0)] <= threshold) * 100

    for threshold_id in range(len(pckh_threshold_range)):
        threshold = pckh_threshold_range[threshold_id]
        joint_distance = distances[:, 8:16]
        pckh[threshold_id, num_keypoints] = np.mean(joint_distance[np.where(joint_distance >= 0)] <= threshold) * 100

    for threshold_id in range(len(pckh_threshold_range)):
        threshold = pckh_threshold_range[threshold_id]
        joints_index = list(range(0, 6)) + list(range(8, 16))
        joint_distance = distances[:, joints_index]
        pckh[threshold_id, num_keypoints + 1] = np.mean(joint_distance[np.where(joint_distance >= 0)] <= threshold) * 100

    return pckh


def print_output(pckh, method_name):
    template = '{0:10} & {1:6} & {2:6} & {3:6} & {4:6} & {5:6} & {6:6} & {7:6} & {8:6} & {9:6}'
    header = template.format('PCKh@0.5', 'Head', 'Sho.', 'Elb.', 'Wri.', 'Hip', 'Knee', 'Ank.', 'U.Body', 'Avg.')
    pckh = template.format(method_name, '%1.2f' % ((pckh[8]  + pckh[9])  / 2),
                                        '%1.2f' % ((pckh[12] + pckh[13]) / 2),
                                        '%1.2f' % ((pckh[11] + pckh[14]) / 2),
                                        '%1.2f' % ((pckh[10] + pckh[15]) / 2),
                                        '%1.2f' % ((pckh[2]  + pckh[3])  / 2),
                                        '%1.2f' % ((pckh[1]  + pckh[4])  / 2),
                                        '%1.2f' % ((pckh[0]  + pckh[5])  / 2),
                                        '%1.2f' % (pckh[-2]),
                                        '%1.2f' % (pckh[-1]))
    print(header)
    print(pckh)


def calc_pckh(gt_path, prediction_path, method_name='gccpm', eval_num=10000):
    threshold_range = np.array([0.5])

    prediction, _ = read_data(prediction_path, False)
    prediction = prediction[:eval_num, :, :]
    gt, _ = read_data(gt_path, True)
    gt = gt[:eval_num, :, :]
    assert gt.shape[0] == prediction.shape[0], 'number of images not matched'
    assert gt.shape[1] == prediction.shape[1], 'number of joints not matched'
    assert gt.shape[2] == prediction.shape[2], 'keypoint dims not matched'

    head_size = get_head_size(gt)
    normalized_distance = get_normalized_distance(gt, prediction, head_size)
    pckh = compute_pckh(normalized_distance, threshold_range)
    print_output(pckh[-1], method_name)

    return pckh

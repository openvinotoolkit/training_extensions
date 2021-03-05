#!/usr/bin/env python2
#
# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from __future__ import print_function

from os import walk
from os.path import join, exists
from argparse import ArgumentParser

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class KMeans(object):
    """Presents algorithm to carry out KMeans clusterization over the prior boxes.
    """

    def __init__(self, train_data, test_data, num_clusters, num_iters=10, eps=1e-5, min_cluster_size=50):
        """Constructor.

        :param train_data: Training data
        :param test_data: Testing data
        :param num_clusters: Number of clusters
        :param num_iters: Max Number of iterations
        :param eps: Epsilon to control minimal changes value
        :param min_cluster_size: Minimal size of cluster to keep it
        """

        self.num_clusters = num_clusters
        self.num_iters = num_iters
        assert self.num_iters is not None and self.num_iters > 0

        self.train_data = train_data
        self.test_data = test_data

        self.eps = eps
        self.centers = None
        self.train_labels = None
        self.test_labels = None
        self.min_cluster_size = min_cluster_size

    @staticmethod
    def _iou(sample, candidates):
        """Calculates IoU metric between all pairs of sample and specified set.

        :param sample: Sample box
        :param candidates: Set of boxes
        :return: List of IoU values
        """

        sample_half_h = 0.5 * sample[0]
        sample_half_w = 0.5 * sample[1]
        sample_bbox = np.array([-sample_half_w, -sample_half_h,
                                sample_half_w, sample_half_h], dtype=np.float32)
        sample_bbox_size = sample[0] * sample[1]

        candidates_half_h = 0.5 * candidates[:, 0]
        candidates_half_w = 0.5 * candidates[:, 1]
        candidates_bbox = [-candidates_half_w, -candidates_half_h,
                           candidates_half_w, candidates_half_h]
        candidates_bbox_size = candidates[:, 0] * candidates[:, 1]

        inter_bbox = [np.maximum(sample_bbox[0], candidates_bbox[0]),
                      np.maximum(sample_bbox[1], candidates_bbox[1]),
                      np.minimum(sample_bbox[2], candidates_bbox[2]),
                      np.minimum(sample_bbox[3], candidates_bbox[3])]
        inter_size = (inter_bbox[2] - inter_bbox[0]) * (inter_bbox[3] - inter_bbox[1])

        iou = inter_size / (sample_bbox_size + candidates_bbox_size - inter_size)

        return iou

    @staticmethod
    def _estimate_data_borders(data):
        """Estimates max and min valid values of input data.

        :param data: Input data
        :return: Tuple of min and max values
        """

        min_values = np.percentile(data, [25.0], axis=0)[0]
        max_values = np.percentile(data, [75.0], axis=0)[0]
        return min_values, max_values

    @staticmethod
    def _enhanced_centers(all_data, num_centers, metric, num_samples=1000, alpha=0.8, min_dist=0.0):
        """Initializes centers by enhanced algorithm.

        :param all_data: Data
        :param num_centers: Number of centers to init
        :param metric: Type of metric
        :param num_samples: Number of samples to carry out on
        :param alpha: Algorithm parameter
        :param min_dist: Min distance to preserve new centers
        :return: List of centers
        """

        num_samples = np.minimum(num_samples, all_data.shape[0])
        rand_idx = np.random.randint(all_data.shape[0], size=num_samples)
        data_subset = all_data[rand_idx, :]
        size_thr = int(float(num_samples) / float(num_centers))

        distances = np.empty([data_subset.shape[0], data_subset.shape[0]], dtype=np.float32)
        for i in xrange(data_subset.shape[0]):
            data_point = data_subset[i]
            distances[i] = metric(data_point, data_subset)
            distances[i, :(i + 1)] = 0.

        factor = 1. / (1. - alpha)
        point_metrics = np.maximum(0., factor * (distances - alpha))

        points_weights = np.mean(point_metrics, axis=1)
        points = sorted([(i, points_weights[i]) for i in xrange(data_subset.shape[0])],
                        key=lambda tup: tup[1], reverse=True)

        init_centers = []
        for point in points:
            if len(init_centers) == 0:
                init_centers.append([point[0], [point[0]], point[1]])
            else:
                max_dist = 0
                best_center_id = None
                for j in xrange(len(init_centers)):
                    cur_distance = distances[point[0], init_centers[j][0]]

                    if cur_distance > max_dist:
                        max_dist = cur_distance
                        best_center_id = j

                if max_dist > min_dist and\
                   best_center_id is not None and\
                   len(init_centers[best_center_id][1]) < size_thr:
                    init_centers[best_center_id][1].append(point[0])
                    init_centers[best_center_id][2] += point[1]
                else:
                    init_centers.append([point[0], [point[0]], point[1]])
        print('Number of init centers: {}'.format(len(init_centers)))

        centers_nodes = sorted([(i, init_centers[i][2] / float(len(init_centers[i][1])))
                                for i in xrange(len(init_centers))], key=lambda tup: tup[1], reverse=True)

        centers = [data_subset[init_centers[centers_nodes[i][0]][0]] for i in xrange(num_centers)]
        centers = np.vstack(centers)

        return centers

    @staticmethod
    def _cluster_points(data, centers, metric):
        """Carry out clustering stage of KMeans algorithm.

        :param data: Input data
        :param centers: Current centers
        :param metric: Type of metric to compare data points
        :return: Clustered data points
        """

        def _init_clusters(num_clusters):
            """Creates list of empty centers.

            :param num_clusters: Target number of centers.
            :return: List of init centers
            """

            init_clusters = []
            for _ in xrange(num_clusters):
                init_clusters.append([])
            return init_clusters

        clusters = _init_clusters(centers.shape[0])
        for i in xrange(data.shape[0]):
            data_point = data[i]
            distances = metric(data_point, centers)
            best_center_id = np.argmax(distances)
            clusters[best_center_id].append(data_point)
        return clusters

    @staticmethod
    def _estimate_clusters_dist(clusters, centers, metric, calc_mean=True, ignore_empty=True):
        """Calculates distances to cluster centers.

        :param clusters: Current clusters
        :param centers: Current centers
        :param metric: Type of metric
        :param calc_mean: Whether to calculate the mean cluster distance instead
        :param ignore_empty: Whether to ignore empty clusters
        :return: List of distances
        """

        num_clusters = len(clusters)
        assert num_clusters == centers.shape[0]

        cluster_distances = []
        for cluster_id in xrange(num_clusters):
            if len(clusters[cluster_id]) > 0:
                cluster_data = clusters[cluster_id]
                all_distances = metric(centers[cluster_id], np.vstack(cluster_data))
                cluster_distances.append(np.median(all_distances))
            else:
                if not ignore_empty:
                    cluster_distances.append(0.)

        if calc_mean:
            return np.min(cluster_distances)
        else:
            return cluster_distances

    @staticmethod
    def _reevaluate_centers(clusters, min_values, max_values, min_cluster_size):
        """Carry out reevaluating stage of KMeans algorithm.

        :param clusters: Current clusters
        :param min_values: Min possible value of new centers
        :param max_values: Max possible value of new centers
        :param min_cluster_size: Minimal size of cluster to preserve it
        :return: List of new cluster centers
        """

        new_centers = []
        for cluster_id in xrange(len(clusters)):
            if len(clusters[cluster_id]) >= min_cluster_size:
                cluster = np.vstack(clusters[cluster_id])
                new_center = np.median(cluster[:, :2], axis=0)
            else:
                print('Created new random center')
                new_center = np.random.uniform(min_values, max_values)
            new_centers.append(new_center)
        return np.vstack(new_centers)

    @staticmethod
    def _estimate_converge(old_centers, new_centers):
        """Calculates current delta of center changes.

        :param old_centers: List of previous step centers
        :param new_centers: List of current centers
        :return: Delta value
        """

        diff = np.abs(old_centers - new_centers)
        distances = np.max(diff, axis=1)
        mean_distance = np.mean(distances)
        return mean_distance

    @staticmethod
    def _fit_data(data, centers, metric):
        """Estimates cluster Ids for the specified data.

        :param data: Input data
        :param centers: Current cluster centers
        :param metric: Type of metric
        :return: Estimated cluster IDs
        """

        labels = np.zeros([data.shape[0]], dtype=np.int32)
        for i in xrange(data.shape[0]):
            data_point = data[i]
            distances = metric(data_point, centers)
            best_center_id = np.argmax(distances)
            labels[i] = best_center_id

        return labels

    def find_centers(self):
        """Carry out KMeans clustering algorithm.
        """

        min_values, max_values = self._estimate_data_borders(self.train_data[:, :2])
        centers = self._enhanced_centers(self.train_data[:, :2], self.num_clusters, self._iou, num_samples=20000)

        best_centers = None
        best_step_id = None

        print('\nKMeans iterations:')
        for i in xrange(self.num_iters):
            old_centers = centers
            clusters = self._cluster_points(self.train_data, centers, self._iou)
            centers = self._reevaluate_centers(clusters, min_values, max_values, self.min_cluster_size)

            center_distances = self._estimate_converge(old_centers, centers)

            train_clusters = self._cluster_points(self.train_data, centers, self._iou)
            train_cluster_distances = self._estimate_clusters_dist(train_clusters, centers, self._iou)

            test_clusters = self._cluster_points(self.test_data, centers, self._iou)
            test_cluster_distances = self._estimate_clusters_dist(test_clusters, centers, self._iou)

            best_centers = centers
            best_step_id = i

            print('   #{}: dist = {:.08f}; train mIoU = {:.05f}%; test mIoU = {:.05f}%'
                  .format(i, center_distances,
                          train_cluster_distances * 100.,
                          test_cluster_distances * 100.))

            if center_distances < self.eps:
                break
        print('\nFinished. Best step: {}'.format(best_step_id))

        train_clusters = self._cluster_points(self.train_data, best_centers, self._iou)
        train_cluster_distances = self._estimate_clusters_dist(train_clusters, best_centers, self._iou,
                                                               calc_mean=False, ignore_empty=False)

        test_clusters = self._cluster_points(self.test_data, best_centers, self._iou)
        test_cluster_distances = self._estimate_clusters_dist(test_clusters, best_centers, self._iou,
                                                              calc_mean=False, ignore_empty=False)

        assert len(train_clusters) == len(test_clusters)

        self.centers = best_centers
        self.train_labels = self._fit_data(self.train_data, best_centers, self._iou)
        self.test_labels = self._fit_data(self.test_data, best_centers, self._iou)

        print('\nEstimated centers:')
        size_factor = 100. / self.train_data.shape[0]
        for i in xrange(best_centers.shape[0]):
            bbox_h = best_centers[i, 0]
            bbox_w = best_centers[i, 1]

            print('   #{}: {}; size = {:.03f}%; train mIoU = {:.05f}%; test mIoU = {:.05f}%'
                  .format(i, [bbox_h, bbox_w],
                          size_factor * float(len(train_clusters[i])),
                          train_cluster_distances[i] * 100.,
                          test_cluster_distances[i] * 100.))

    def plot_data(self):
        """Plots data points.
        """

        assert self.centers is not None
        assert self.train_labels is not None

        num_clusters = self.centers.shape[0]
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, num_clusters)]  # pylint: disable=no-member

        for k, col in zip(xrange(num_clusters), colors):
            my_members = self.train_labels == k
            plt.plot(self.train_data[my_members, 0], self.train_data[my_members, 1], 'w',
                     markerfacecolor=col, marker='.', alpha=0.7)

        for k, col in zip(xrange(num_clusters), colors):
            cluster_center = self.centers[k]
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        plt.show()


def load_data(root_dir_path, extension='.json'):
    """Loads data points.

    :param root_dir_path: Path to start data loading
    :param extension: Valid file extensions
    :return: Loaded data
    """

    list_of_files = []
    for dir_path, _, file_names in walk(root_dir_path):
        list_of_files += [join(dir_path, file_name) for file_name in file_names if file_name.endswith(extension)]

    out_data = []
    for file_path in tqdm(list_of_files, desc='Loading data'):
        with open(file_path, 'r') as read_file:
            bboxes = json.load(read_file)

        for bbox in bboxes:
            bbox_h = bbox['ymax'] - bbox['ymin']
            bbox_w = bbox['xmax'] - bbox['xmin']
            out_data.append((bbox_h, bbox_w))

    return np.array(out_data, dtype=np.float32)


def filter_data(bbox_data, enable=True, ar_low_percentile=0.1, ar_top_percentile=99.9, data_fraction=1.0,
                min_height=None, max_height=None):
    """Filter data if it's too large.

    :param bbox_data: Input data
    :param enable: Whether to enable filtering
    :param ar_low_percentile: Min aspect ration percentile
    :param ar_top_percentile: Max aspect ration percentile
    :param data_fraction: Fraction of data to preserve
    :param min_height: Min box height
    :param max_height: Max box height
    :return: Filtered data
    """

    if enable:
        aspect_ratios = bbox_data[:, 0] / bbox_data[:, 1]
        low_border = np.percentile(aspect_ratios, [float(ar_low_percentile)], axis=0, keepdims=True)[0]
        top_border = np.percentile(aspect_ratios, [float(ar_top_percentile)], axis=0, keepdims=True)[0]
        mask = np.logical_and(aspect_ratios > low_border, aspect_ratios < top_border)
        filtered_bbox_data = bbox_data[mask]

        if min_height is not None:
            filtered_bbox_data = bbox_data[bbox_data[:, 0] > min_height]

        if max_height is not None:
            filtered_bbox_data = bbox_data[bbox_data[:, 0] < max_height]

        if data_fraction is not None and data_fraction < 1.0:
            out_train_data_size = int(data_fraction * filtered_bbox_data.shape[0])
            assert out_train_data_size > 0

            idx = np.random.randint(filtered_bbox_data.shape[0], size=out_train_data_size)
            filtered_bbox_data = filtered_bbox_data[idx]

        print('Filtered data size: {}'.format(len(filtered_bbox_data)))

        return filtered_bbox_data
    else:
        return bbox_data


def print_data_stat(data, name):
    """Prints input data statistics.

    :param data: Input data
    :param name: Header name
    """

    bbox_heights = data[:, 0]
    aspect_ratios = data[:, 0] / data[:, 1]
    print('{} data: {} samples\n'
          '   aspect ratios: min = {}, median = {}, max = {}\n'
          '   height: min = {}, median = {}, max = {}'
          .format(name, data.shape[0],
                  np.min(aspect_ratios), np.median(aspect_ratios), np.max(aspect_ratios),
                  np.min(bbox_heights), np.median(bbox_heights), np.max(bbox_heights)))


def main():
    """Main function.
    """

    parser = ArgumentParser()
    parser.add_argument('--train_path', '-t', type=str, required=True, help='Path to directory with annotation files')
    parser.add_argument('--val_path', '-v', type=str, required=True, help='Path to directory with annotation files')
    parser.add_argument('--num_clusters', '-n', type=int, required=True, help='Number of clusters')
    parser.add_argument('--num_iters', '-i', type=int, required=False, default=50, help='Number of iterations')
    parser.add_argument('--train_fraction', type=float, required=False, default=0.5, help='Fraction of train subset')
    parser.add_argument('--val_fraction', type=float, required=False, default=0.8, help='Fraction of val subset')
    parser.add_argument('--min_height', type=float, required=False, default=None, help='Min box height')
    parser.add_argument('--max_height', type=float, required=False, default=None, help='Max box height')
    args = parser.parse_args()

    assert exists(args.train_path)
    assert exists(args.val_path)
    assert args.num_clusters > 0
    assert args.num_iters > 0

    train_data = filter_data(load_data(args.train_path), data_fraction=args.train_fraction,
                             min_height=args.min_height, max_height=args.max_height)
    test_data = filter_data(load_data(args.val_path), data_fraction=args.val_fraction,
                            min_height=args.min_height, max_height=args.max_height)

    print_data_stat(train_data, 'Train')
    print_data_stat(test_data, 'Test')

    kmeans = KMeans(train_data, test_data, num_clusters=args.num_clusters, num_iters=args.num_iters)
    kmeans.find_centers()
    kmeans.plot_data()


if __name__ == '__main__':
    main()

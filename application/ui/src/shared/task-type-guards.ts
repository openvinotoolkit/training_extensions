// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Task, TaskType } from '@/api/types';

export const isClassificationTask = (taskType: TaskType | null): boolean => {
    return taskType === 'classification';
};

export const isDetectionTask = (taskType: TaskType | null): boolean => {
    return taskType === 'detection';
};

export const isSegmentationTask = (taskType: TaskType | null): boolean => {
    return taskType === 'instance_segmentation';
};

export const isMultiLabelClassificationTask = (task: Task): boolean => {
    return isClassificationTask(task.task_type) && task.exclusive_labels === false;
};

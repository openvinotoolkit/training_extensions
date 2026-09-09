// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Task, TaskType } from '@/api/types';
import dayjs from 'dayjs';
import type { TFunction } from 'i18next';

import { isMultiLabelClassificationTask } from '../task-type-guards';

export const formatCreationDate = (creationDate: string) => {
    return dayjs(creationDate).format('D MMMM YYYY | h:mm A');
};

export const MAP_PROJECT_TYPE_TO_TITLE_KEY = {
    detection: 'project.taskTypes.detection',
    classification: 'project.taskTypes.classification',
    instance_segmentation: 'project.taskTypes.instanceSegmentation',
} as const satisfies Record<TaskType, string>;

export const getProjectTypeTitle = (task: Task | undefined, t: TFunction): string | undefined => {
    if (task === undefined) {
        return undefined;
    }

    return isMultiLabelClassificationTask(task)
        ? t('project.taskTypes.multiLabelClassification')
        : t(MAP_PROJECT_TYPE_TO_TITLE_KEY[task.task_type]);
};

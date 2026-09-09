// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import type { Label, TaskType } from '@/api/types';
import { useProject } from 'hooks/api/project.hook';
import { negate } from 'lodash-es';

import { isClassificationTask } from '../../features/project/task-type-guards';
import type { AnnotationLabel, AnnotationLabelRef } from '../types';

export const EMPTY_LABEL_ID = 'empty-label';
const NO_LABEL: Label = { id: EMPTY_LABEL_ID, name: 'No label', color: 'var(--no-label)', hotkey: 'N' };
const NO_OBJECT_LABEL: Label = { id: EMPTY_LABEL_ID, name: 'No object', color: 'var(--no-label)', hotkey: 'N' };

export const isEmptyLabel = <T extends { id: string }>({ id }: T): boolean => id === EMPTY_LABEL_ID;
export const isNonEmptyLabel = negate(isEmptyLabel);

const getEmptyLabel = (taskType: TaskType, exclusiveLabels: boolean): Label | null => {
    if (isClassificationTask(taskType)) {
        const isMultiLabel = exclusiveLabels === false;

        if (isMultiLabel) {
            return NO_LABEL;
        }

        return null;
    }

    return NO_OBJECT_LABEL;
};

export const useProjectLabelsWithEmptyLabel = (): Label[] => {
    const { data: project } = useProject();
    const { labels = [], exclusive_labels, task_type } = project.task;

    return useMemo(() => {
        const label = getEmptyLabel(task_type, exclusive_labels);
        if (label) {
            return [...labels, label];
        }

        return labels;
    }, [exclusive_labels, labels, task_type]);
};

export const filterOutEmptyLabels = <T extends Pick<Label, 'id'>>(labels: T[]): T[] =>
    labels.filter((label) => label.id !== EMPTY_LABEL_ID);

export const useLabelResolver = () => {
    const labels = useProjectLabelsWithEmptyLabel();

    const labelMap = useMemo(() => new Map(labels.map((label) => [label.id, label])), [labels]);

    const getLabel = (id: string): Label | undefined => labelMap.get(id);

    const resolveAnnotationLabel = (ref: AnnotationLabelRef): AnnotationLabel | undefined => {
        const label = labelMap.get(ref.id);

        if (label === undefined) {
            return undefined;
        }

        return ref.probability !== undefined ? { ...label, probability: ref.probability } : label;
    };

    return { getLabel, resolveAnnotationLabel };
};

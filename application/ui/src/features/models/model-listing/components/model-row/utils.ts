// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Evaluation, Metric, Model, TaskType } from '@/api/types';
import type { TranslateFn } from '@/i18n';

import { isClassificationTask } from '../../../../project/task-type-guards';

export const getModelEvaluations = (model: Model): Evaluation[] => {
    return model.variants.flatMap((variant) => variant.evaluations);
};

const getDefaultPerformanceMetricName = (taskType: TaskType | null): string => {
    return isClassificationTask(taskType) ? 'Accuracy' : 'mAP';
};

const getTestingEvaluation = (evaluations: Evaluation[]): Evaluation | undefined => {
    return evaluations.find(({ subset }) => subset === 'testing');
};

export const getTestingMetrics = (evaluations: Evaluation[]): Metric[] => {
    return getTestingEvaluation(evaluations)?.metrics ?? [];
};

export const getTestingMetric = (model: Model): { name: string; value: number } | undefined => {
    const primaryMetric = getTestingMetrics(getModelEvaluations(model)).find(({ primary }) => primary);

    if (primaryMetric !== undefined) {
        return { name: primaryMetric.name, value: Math.round(primaryMetric.value * 100) };
    }

    return undefined;
};

export const getFirstAvailableTestingMetric = (
    models: Model[] | undefined
): { name: string; value: number } | undefined => {
    // Should never happen, but just in case
    if (models === undefined) {
        return undefined;
    }

    for (const model of models) {
        const testingMetric = getTestingMetric(model);

        if (testingMetric !== undefined) {
            return testingMetric;
        }
    }

    return undefined;
};

export const getPerformanceColumnAriaLabel = (models: Model[] | undefined, taskType: TaskType | null): string => {
    return getFirstAvailableTestingMetric(models)?.name ?? getDefaultPerformanceMetricName(taskType);
};

export const getPerformanceColumnLabel = (
    models: Model[] | undefined,
    taskType: TaskType | null,
    t: TranslateFn
): string => {
    return (
        getFirstAvailableTestingMetric(models)?.name ??
        (isClassificationTask(taskType) ? t('models.performance.accuracy') : 'mAP')
    );
};

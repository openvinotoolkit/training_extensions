// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '@/api/types';
import type { TranslateFn } from '@/i18n';
import { orderBy } from 'lodash-es';

export const SortingOptions = {
    NAME_ASC: 'name-asc',
    NAME_DESC: 'name-desc',
    SPEED_ASC: 'speed-asc',
    SPEED_DESC: 'speed-desc',
    ACCURACY_ASC: 'accuracy-asc',
    ACCURACY_DESC: 'accuracy-desc',
} as const;

export type SortingOptions = (typeof SortingOptions)[keyof typeof SortingOptions];

type SortingHandler = (
    modelArchitectures: ModelArchitectureWithPerformanceCategory[]
) => ModelArchitectureWithPerformanceCategory[];

const getAccuracyMetricBasedOnTask = ({ stats }: ModelArchitectureWithPerformanceCategory) => {
    const benchmarkMetrics = stats?.benchmark_metrics;

    return (
        benchmarkMetrics?.imagenet_top1_accuracy ?? benchmarkMetrics?.coco_map_50_95 ?? benchmarkMetrics?.coco_map_50
    );
};

export const SORTING_HANDLERS: Record<SortingOptions, SortingHandler> = {
    [SortingOptions.ACCURACY_ASC]: (modelArchitectures) =>
        orderBy(modelArchitectures, getAccuracyMetricBasedOnTask, 'asc'),
    [SortingOptions.ACCURACY_DESC]: (modelArchitectures) =>
        orderBy(modelArchitectures, getAccuracyMetricBasedOnTask, 'desc'),
    [SortingOptions.NAME_ASC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.name, 'asc'),
    [SortingOptions.NAME_DESC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.name, 'desc'),
    [SortingOptions.SPEED_ASC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.stats?.gigaflops, 'asc'),
    [SortingOptions.SPEED_DESC]: (modelArchitectures) =>
        orderBy(modelArchitectures, (modelArchitecture) => modelArchitecture.stats?.gigaflops, 'desc'),
};

export const getSortOptions = (t: TranslateFn) => [
    [
        {
            key: SortingOptions.NAME_ASC,
            name: t('models.training.architectures.sort.nameAsc'),
        },
        {
            key: SortingOptions.NAME_DESC,
            name: t('models.training.architectures.sort.nameDesc'),
        },
    ],
    [
        {
            key: SortingOptions.SPEED_ASC,
            name: t('models.training.architectures.sort.speedAsc'),
        },
        {
            key: SortingOptions.SPEED_DESC,
            name: t('models.training.architectures.sort.speedDesc'),
        },
    ],
    [
        {
            key: SortingOptions.ACCURACY_ASC,
            name: t('models.training.architectures.sort.accuracyAsc'),
        },
        {
            key: SortingOptions.ACCURACY_DESC,
            name: t('models.training.architectures.sort.accuracyDesc'),
        },
    ],
];

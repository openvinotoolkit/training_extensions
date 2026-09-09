// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '@/api/types';
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

export const SORT_OPTIONS = [
    [
        {
            key: SortingOptions.NAME_ASC,
            name: 'Name (A to Z)',
        },
        {
            key: SortingOptions.NAME_DESC,
            name: 'Name (Z to A)',
        },
    ],
    [
        {
            key: SortingOptions.SPEED_ASC,
            name: 'Speed (fastest first)',
        },
        {
            key: SortingOptions.SPEED_DESC,
            name: 'Speed (slowest first)',
        },
    ],
    [
        {
            key: SortingOptions.ACCURACY_ASC,
            name: 'Accuracy (lowest first)',
        },
        {
            key: SortingOptions.ACCURACY_DESC,
            name: 'Accuracy (highest first)',
        },
    ],
];

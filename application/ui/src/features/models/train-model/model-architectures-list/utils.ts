// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { BenchmarkMetrics, ModelArchitectureWithPerformanceCategory } from '@/api/types';
import { isNil } from 'lodash-es';

type AccuracyMetric = { label: string; value: number };

type BenchmarkMetricKey = keyof BenchmarkMetrics;

const ACCURACY_METRIC_LABELS: Partial<Record<BenchmarkMetricKey, string>> = {
    imagenet_top1_accuracy: 'Top-1 Acc on ImageNet',
    coco_map_50_95: 'mAP on COCO',
    coco_map_50: 'mAP50 on COCO',
};

export const getAccuracyMetric = (
    modelArchitecture: ModelArchitectureWithPerformanceCategory
): AccuracyMetric | undefined => {
    const benchmarkMetrics = modelArchitecture.stats?.benchmark_metrics;

    for (const [key, label] of Object.entries(ACCURACY_METRIC_LABELS)) {
        const value = benchmarkMetrics?.[key as BenchmarkMetricKey];

        if (!isNil(value)) {
            return { label, value };
        }
    }

    return undefined;
};

export const getRecommendedArchitectures = (modelArchitectures: ModelArchitectureWithPerformanceCategory[]) => {
    const recommended = modelArchitectures.filter(
        (modelArchitecture) => modelArchitecture.performanceCategory !== undefined
    );

    if (recommended.length > 0) {
        return recommended;
    }

    // For now just return top 3 recommended architectures, but in the future we can add more logic here
    return modelArchitectures.slice(0, 3);
};

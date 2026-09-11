// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelVariant } from '@/api/types';
import type { TranslateFn } from '@/i18n';

import { getTestingMetrics } from '../components/model-row/utils';

type PrimaryTestingMetricValue = {
    name: string;
    value: number;
};

const getPrimaryTestingMetric = (variant: ModelVariant) => {
    return getTestingMetrics(variant.evaluations).find(({ primary }) => primary);
};

export const getPrimaryTestingMetricValue = (
    variant: ModelVariant | undefined
): PrimaryTestingMetricValue | undefined => {
    const primaryMetric = variant ? getPrimaryTestingMetric(variant) : undefined;

    if (primaryMetric === undefined) {
        return undefined;
    }

    return { name: primaryMetric.name, value: Math.round(primaryMetric.value * 100) };
};

export const getFp32PytorchVariant = (variants: ModelVariant[]): ModelVariant | undefined => {
    return variants.find((variant) => variant.format === 'pytorch' && variant.precision === 'fp32');
};

export const getBaselineVariant = (variants: ModelVariant[]): ModelVariant | undefined => {
    return (
        variants.find((variant) => variant.precision === 'fp16') ??
        variants.find((variant) => variant.precision === 'fp32')
    );
};

export const getPerformanceColumnName = (
    variants: ModelVariant[],
    fp32PytorchMetric: PrimaryTestingMetricValue | undefined,
    t: TranslateFn
): string => {
    return (
        variants.map((variant) => getPrimaryTestingMetricValue(variant)).find((metric) => metric !== undefined)?.name ??
        fp32PytorchMetric?.name ??
        t('models.performance.accuracy')
    );
};

export const getVariantPerformanceValue = (
    variant: ModelVariant,
    fp32PytorchMetric: PrimaryTestingMetricValue | undefined
): number | undefined => {
    const metric = getPrimaryTestingMetricValue(variant);

    if (metric !== undefined) {
        return metric.value;
    }

    if ((variant.evaluations?.length ?? 0) === 0) {
        return fp32PytorchMetric?.value;
    }

    return undefined;
};

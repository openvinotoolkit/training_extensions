// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createI18nInstance } from '@/i18n';
import { getMockedVariant } from 'mocks/mock-model-variant';

import {
    getFp32PytorchVariant,
    getPerformanceColumnName,
    getPrimaryTestingMetricValue,
    getVariantPerformanceValue,
} from './variant-metrics';

describe('variant-metrics utilities', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    it('returns primary testing metric value rounded to percent', () => {
        const variant = getMockedVariant({
            evaluations: [
                {
                    dataset_revision_id: 'dataset-1',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.923, primary: true }],
                },
            ],
        });

        expect(getPrimaryTestingMetricValue(variant)).toEqual({ name: 'Accuracy', value: 92 });
    });

    it('returns undefined when variant is undefined or has no primary testing metric', () => {
        const variant = getMockedVariant({
            evaluations: [
                {
                    dataset_revision_id: 'dataset-1',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.9, primary: false }],
                },
            ],
        });

        expect(getPrimaryTestingMetricValue(undefined)).toBeUndefined();
        expect(getPrimaryTestingMetricValue(variant)).toBeUndefined();
    });

    it('returns fp32 pytorch variant when available', () => {
        const variants = [
            getMockedVariant({ id: 'ov-1', format: 'openvino', precision: 'fp16' }),
            getMockedVariant({ id: 'pt-1', format: 'pytorch', precision: 'fp32' }),
        ];

        expect(getFp32PytorchVariant(variants)?.id).toBe('pt-1');
    });

    it('returns performance column name from variants, then fp32 fallback, then Accuracy', () => {
        const withMetric = [
            getMockedVariant({
                evaluations: [
                    {
                        dataset_revision_id: 'dataset-1',
                        subset: 'testing',
                        metrics: [{ name: 'mAP', value: 0.91, primary: true }],
                    },
                ],
            }),
        ];

        expect(getPerformanceColumnName(withMetric, undefined, t)).toBe('mAP');
        expect(getPerformanceColumnName([], { name: 'Accuracy', value: 87 }, t)).toBe('Accuracy');
        expect(getPerformanceColumnName([], undefined, t)).toBe('Accuracy');
    });

    it('returns variant performance and applies fallback only for empty evaluations', () => {
        const fp32Metric = { name: 'mAP', value: 87 };
        const withMetric = getMockedVariant({
            evaluations: [
                {
                    dataset_revision_id: 'dataset-1',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.89, primary: true }],
                },
            ],
        });
        const emptyEvaluations = getMockedVariant({ evaluations: [] });
        const noPrimaryButHasEvaluations = getMockedVariant({
            evaluations: [
                {
                    dataset_revision_id: 'dataset-1',
                    subset: 'testing',
                    metrics: [{ name: 'Accuracy', value: 0.9, primary: false }],
                },
            ],
        });

        expect(getVariantPerformanceValue(withMetric, fp32Metric)).toBe(89);
        expect(getVariantPerformanceValue(emptyEvaluations, fp32Metric)).toBe(fp32Metric.value);
        expect(getVariantPerformanceValue(noPrimaryButHasEvaluations, fp32Metric)).toBeUndefined();
    });
});

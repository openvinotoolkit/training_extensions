// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createI18nInstance } from '@/i18n';

import { getMockedModelArchitecture } from '../../../../../mocks/mock-model';
import { getAccuracyMetric, getRecommendedArchitectures } from './utils';

describe('getRecommendedArchitectures', () => {
    it('returns recommended architectures when performanceCategory is defined', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1', performanceCategory: 'balance' }),
            getMockedModelArchitecture({ id: 'arch-2', performanceCategory: 'speed' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
        ];

        const result = getRecommendedArchitectures(modelArchitectures);

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
    });

    it('returns top 3 architectures when no performanceCategory is defined', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1' }),
            getMockedModelArchitecture({ id: 'arch-2' }),
            getMockedModelArchitecture({ id: 'arch-3' }),
            getMockedModelArchitecture({ id: 'arch-4' }),
        ];

        const result = getRecommendedArchitectures(modelArchitectures);

        expect(result).toHaveLength(3);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
        expect(result[2].id).toBe('arch-3');
    });

    it('handles empty model architectures array', () => {
        const result = getRecommendedArchitectures([]);

        expect(result).toHaveLength(0);
    });

    it('handles model architectures with less than 3 items and no performanceCategory', () => {
        const modelArchitectures = [
            getMockedModelArchitecture({ id: 'arch-1' }),
            getMockedModelArchitecture({ id: 'arch-2' }),
        ];

        const result = getRecommendedArchitectures(modelArchitectures);

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe('arch-1');
        expect(result[1].id).toBe('arch-2');
    });
});

describe('getAccuracyMetric', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    it('returns Top-1 Acc for classification tasks', () => {
        const modelArchitecture = getMockedModelArchitecture({
            task: 'classification',
            stats: {
                gigaflops: 1,
                trainable_parameters: 5,
                benchmark_metrics: {
                    imagenet_top1_accuracy: 76.2,
                    imagenet_top5_accuracy: 95.3,
                    coco_map_50_95: null,
                    coco_map_50: null,
                },
            },
        });

        const result = getAccuracyMetric(modelArchitecture, t);

        expect(result).toEqual({ label: 'Top-1 Acc on ImageNet', value: 76.2 });
    });

    it('returns mAP (50-95) for detection tasks', () => {
        const modelArchitecture = getMockedModelArchitecture({
            task: 'detection',
            stats: {
                gigaflops: 91,
                trainable_parameters: 31,
                benchmark_metrics: {
                    imagenet_top1_accuracy: null,
                    imagenet_top5_accuracy: null,
                    coco_map_50_95: 55.3,
                    coco_map_50: 72.1,
                },
            },
        });

        const result = getAccuracyMetric(modelArchitecture, t);

        expect(result).toEqual({ label: 'mAP on COCO', value: 55.3 });
    });

    it('falls back to coco_map_50 when coco_map_50_95 is null', () => {
        const modelArchitecture = getMockedModelArchitecture({
            task: 'detection',
            stats: {
                gigaflops: 91,
                trainable_parameters: 31,
                benchmark_metrics: {
                    imagenet_top1_accuracy: null,
                    imagenet_top5_accuracy: null,
                    coco_map_50_95: null,
                    coco_map_50: 72.1,
                },
            },
        });

        const result = getAccuracyMetric(modelArchitecture, t);

        expect(result).toEqual({ label: 'mAP50 on COCO', value: 72.1 });
    });

    it('returns undefined when no accuracy metrics are available', () => {
        const modelArchitecture = getMockedModelArchitecture({
            task: 'detection',
            stats: {
                gigaflops: 91,
                trainable_parameters: 31,
                benchmark_metrics: {
                    imagenet_top1_accuracy: null,
                    imagenet_top5_accuracy: null,
                    coco_map_50_95: null,
                    coco_map_50: null,
                },
            },
        });

        const result = getAccuracyMetric(modelArchitecture, t);

        expect(result).toBeUndefined();
    });
});

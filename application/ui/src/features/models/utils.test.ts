// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedModel } from 'mocks/mock-model';
import { getMockedVariant } from 'mocks/mock-model-variant';

import {
    distributeByLargestRemainder,
    getAllModelsWithOpenVINOVariants,
    getModelIdentifierPayload,
    isEdgeCrafterModel,
    isUltralyticsModel,
    SelectableModel,
} from './utils';

describe('distributeByLargestRemainder', () => {
    it('returns all zeros when total is zero', () => {
        expect(distributeByLargestRemainder([70, 20, 10], 0)).toEqual([0, 0, 0]);
    });

    it('returns all zeros when sum of values is zero', () => {
        expect(distributeByLargestRemainder([0, 0, 0], 100)).toEqual([0, 0, 0]);
    });

    it('returns empty array for empty input with non-zero total', () => {
        expect(distributeByLargestRemainder([], 100)).toEqual([]);
    });

    it('result always sums to total', () => {
        const result = distributeByLargestRemainder([750, 125, 125], 100);
        expect(result.reduce((a, b) => a + b, 0)).toBe(100);
        expect(result).toEqual([75, 13, 12]);
    });

    it('tie-breaking: largest fractional part gets the extra unit', () => {
        // [1, 1, 1] → each gets 33.33%, floors to 33, remainder = 1
        // all fractionals equal → first in sorted order wins
        const result = distributeByLargestRemainder([1, 1, 1], 100);
        expect(result.reduce((a, b) => a + b, 0)).toBe(100);
        expect(result).toEqual([34, 33, 33]);
    });

    it('count case: 45/28/27 split on 5 items → [2, 2, 1]', () => {
        expect(distributeByLargestRemainder([45, 28, 27], 5)).toEqual([2, 2, 1]);
    });

    it('standard percentage case: 70/20/10 on 100 → [70, 20, 10]', () => {
        expect(distributeByLargestRemainder([70, 20, 10], 100)).toEqual([70, 20, 10]);
    });

    it('handles a single value', () => {
        expect(distributeByLargestRemainder([5], 100)).toEqual([100]);
    });
});

describe('getAllModelsWithOpenVINOVariants', () => {
    it('returns empty array for empty models array', () => {
        expect(getAllModelsWithOpenVINOVariants([])).toEqual([]);
    });

    it('returns empty array when model has no variants', () => {
        const model = getMockedModel({ id: 'model-1', name: 'My Model', variants: [] });

        expect(getAllModelsWithOpenVINOVariants([model])).toEqual([]);
    });

    it('returns empty array when model has no openvino variants', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [
                getMockedVariant({ id: 'v-onnx', format: 'onnx', precision: 'fp32' }),
                getMockedVariant({ id: 'v-pytorch', format: 'pytorch', precision: 'fp32' }),
            ],
        });

        expect(getAllModelsWithOpenVINOVariants([model])).toEqual([]);
    });

    it('returns openvino variant with precision uppercased and parent modelId', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [getMockedVariant({ id: 'v-ov', format: 'openvino', precision: 'fp16' })],
        });

        expect(getAllModelsWithOpenVINOVariants([model])).toEqual([
            { modelVariantId: 'v-ov', name: 'My Model [FP16]', modelId: 'model-1', optimalConfidenceThreshold: 0.65 },
        ]);
    });

    it('returns all openvino variants in order', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [
                getMockedVariant({ id: 'v-fp16', format: 'openvino', precision: 'fp16' }),
                getMockedVariant({ id: 'v-fp32', format: 'openvino', precision: 'fp32' }),
                getMockedVariant({ id: 'v-int8', format: 'openvino', precision: 'int8' }),
            ],
        });

        expect(getAllModelsWithOpenVINOVariants([model])).toEqual([
            { modelVariantId: 'v-fp16', name: 'My Model [FP16]', modelId: 'model-1', optimalConfidenceThreshold: 0.65 },
            { modelVariantId: 'v-fp32', name: 'My Model [FP32]', modelId: 'model-1', optimalConfidenceThreshold: 0.65 },
            { modelVariantId: 'v-int8', name: 'My Model [INT8]', modelId: 'model-1', optimalConfidenceThreshold: 0.65 },
        ]);
    });

    it('returns only openvino variants when formats are mixed', () => {
        const model = getMockedModel({
            id: 'model-1',
            name: 'My Model',
            variants: [
                getMockedVariant({ id: 'v-ov', format: 'openvino', precision: 'fp16' }),
                getMockedVariant({ id: 'v-onnx', format: 'onnx', precision: 'fp32' }),
            ],
        });

        expect(getAllModelsWithOpenVINOVariants([model])).toEqual([
            { modelVariantId: 'v-ov', name: 'My Model [FP16]', modelId: 'model-1', optimalConfidenceThreshold: 0.65 },
        ]);
    });

    it('processes multiple models and preserves their order', () => {
        const modelA = getMockedModel({
            id: 'model-a',
            name: 'Model A',
            variants: [getMockedVariant({ id: 'v-a-ov', format: 'openvino', precision: 'fp16' })],
        });
        const modelB = getMockedModel({ id: 'model-b', name: 'Model B', variants: [] });
        const modelC = getMockedModel({
            id: 'model-c',
            name: 'Model C',
            variants: [
                getMockedVariant({ id: 'v-c-fp32', format: 'openvino', precision: 'fp32' }),
                getMockedVariant({ id: 'v-c-int8', format: 'openvino', precision: 'int8' }),
            ],
        });

        expect(getAllModelsWithOpenVINOVariants([modelA, modelB, modelC])).toEqual([
            { modelVariantId: 'v-a-ov', name: 'Model A [FP16]', modelId: 'model-a', optimalConfidenceThreshold: 0.65 },
            {
                modelVariantId: 'v-c-fp32',
                name: 'Model C [FP32]',
                modelId: 'model-c',
                optimalConfidenceThreshold: 0.65,
            },
            {
                modelVariantId: 'v-c-int8',
                name: 'Model C [INT8]',
                modelId: 'model-c',
                optimalConfidenceThreshold: 0.65,
            },
        ]);
    });
});

describe('getModelIdentifierPayload', () => {
    it('returns both model_id (parent) and model_variant_id', () => {
        const model: SelectableModel = {
            modelVariantId: 'v-ov',
            name: 'My Model [FP16]',
            modelId: 'model-1',
            optimalConfidenceThreshold: 0.65,
        };

        expect(getModelIdentifierPayload(model)).toEqual({ model_id: 'model-1', model_variant_id: 'v-ov' });
    });
});

describe('isUltralyticsModel', () => {
    it('returns true for ultralytics model identifier', () => {
        const models = [
            'object-detection-yolo26-m',
            'OBJECT-DETECTION-YOLO26-M',
            'object-detection-yolo11-n',
            'OBJECT-DETECTION-YOLO11-N',
            'object-detection-yolo12-s',
            'OBJECT-DETECTION-YOLO12-S',
            'instance-segmentation-yolo11-l',
            'instance-segmentation-yolo12-x',
        ];

        models.forEach((model) => {
            expect(isUltralyticsModel(model)).toBe(true);
        });
    });

    it('returns false for non ultralytics model identifier', () => {
        expect(isUltralyticsModel('object-detection-yolox-l')).toBe(false);
        expect(isUltralyticsModel('OBJECT-DETECTION-YOLOX-L')).toBe(false);
    });
});

describe('isEdgeCrafterModel', () => {
    it('returns true for edgecrafter model identifier', () => {
        const models = [
            'object-detection-edgecrafter-l',
            'OBJECT-DETECTION-EDGECRAFTER-L',
            'object-detection-edgecrafter-m',
            'object-detection-edgecrafter-s',
            'object-detection-edgecrafter-x',
        ];

        models.forEach((model) => {
            expect(isEdgeCrafterModel(model)).toBe(true);
        });
    });

    it('returns false for non edgecrafter model identifier', () => {
        expect(isEdgeCrafterModel('object-detection-yolox-l')).toBe(false);
        expect(isEdgeCrafterModel('OBJECT-DETECTION-YOLOX-L')).toBe(false);
    });
});

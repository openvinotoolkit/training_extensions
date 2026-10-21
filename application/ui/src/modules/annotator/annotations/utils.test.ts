// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedAnnotationLabelRef } from 'mocks/mock-labels';

import { convertPredictionToAnnotation, isPrediction } from './utils';

describe('isPrediction', () => {
    it('returns true when probability is defined', () => {
        const labelRef = getMockedAnnotationLabelRef({ probability: 0.9 });

        expect(isPrediction(labelRef)).toBe(true);
    });

    it('returns true when probability is 0', () => {
        const labelRef = getMockedAnnotationLabelRef({ probability: 0 });

        expect(isPrediction(labelRef)).toBe(true);
    });

    it('returns false when probability is undefined', () => {
        const labelRef = getMockedAnnotationLabelRef();

        expect(isPrediction(labelRef)).toBe(false);
    });
});

describe('convertPredictionToAnnotation', () => {
    it('removes probability from all prediction label refs', () => {
        const annotation = getMockedAnnotation({
            labels: [
                getMockedAnnotationLabelRef({ probability: 0.95 }),
                getMockedAnnotationLabelRef({ id: 'label-2', probability: 0.7 }),
            ],
        });

        const result = convertPredictionToAnnotation(annotation);

        result.labels.forEach((ref) => {
            expect(ref).not.toHaveProperty('probability');
        });
    });

    it('preserves the id after conversion', () => {
        const labelRef = getMockedAnnotationLabelRef({ id: 'label-1', probability: 0.85 });
        const annotation = getMockedAnnotation({ labels: [labelRef] });

        const result = convertPredictionToAnnotation(annotation);

        expect(result.labels[0]).toEqual({ id: 'label-1' });
    });

    it('preserves annotation shape and id after conversion', () => {
        const annotation = getMockedAnnotation({
            labels: [getMockedAnnotationLabelRef({ probability: 0.6 })],
        });

        const result = convertPredictionToAnnotation(annotation);

        expect(result.id).toBe(annotation.id);
        expect(result.shape).toEqual(annotation.shape);
    });
});

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedAnnotationLabelRef, getMockedLabel } from 'mocks/mock-labels';

import { toggleLabel } from './util';

describe('secondary toolbar utils', () => {
    describe('toggleLabel', () => {
        const mockLabel1 = getMockedLabel({ id: 'label-1', name: 'Label 1' });
        const mockLabel2 = getMockedLabel({ id: 'label-2', name: 'Label 2' });
        const mockLabel3 = getMockedLabel({ id: 'label-3', name: 'Label 3' });

        it('add label when it does not exist in annotation', () => {
            const annotation = getMockedAnnotation({
                labels: [
                    getMockedAnnotationLabelRef({ id: 'label-1' }),
                    getMockedAnnotationLabelRef({ id: 'label-2' }),
                ],
            });

            const result = toggleLabel(mockLabel3, annotation.labels);

            expect(result).toEqual([{ id: 'label-1' }, { id: 'label-2' }, { id: 'label-3' }]);
        });

        it('remove label when it exists in annotation', () => {
            const annotation = getMockedAnnotation({
                labels: [
                    getMockedAnnotationLabelRef({ id: 'label-1' }),
                    getMockedAnnotationLabelRef({ id: 'label-2' }),
                    getMockedAnnotationLabelRef({ id: 'label-3' }),
                ],
            });

            const result = toggleLabel(mockLabel2, annotation.labels);

            expect(result).toEqual([{ id: 'label-1' }, { id: 'label-3' }]);
        });

        it('add label to empty labels array', () => {
            const annotation = getMockedAnnotation({ labels: [] });

            const result = toggleLabel(mockLabel1, annotation.labels);

            expect(result).toEqual([{ id: 'label-1' }]);
        });

        it('remove the only label from annotation', () => {
            const annotation = getMockedAnnotation({
                labels: [getMockedAnnotationLabelRef({ id: 'label-1' })],
            });

            const result = toggleLabel(mockLabel1, annotation.labels);

            expect(result).toEqual([]);
        });
    });
});

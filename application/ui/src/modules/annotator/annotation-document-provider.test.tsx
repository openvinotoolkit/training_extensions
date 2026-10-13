// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import type { AnnotationDTO } from '@/api/types';
import { act } from '@testing-library/react';
import { getMockedShape } from 'mocks/mock-annotation';

import { renderHook } from '../../test-utils/render';
import {
    AnnotationDocumentProvider,
    useAnnotationCommands,
    useAnnotations,
    useIsAnnotatorReadOnly,
    type AnnotationDocumentProviderProps,
} from './annotation-document-provider.component';

const renderDocument = ({
    initialAnnotationsDTO = [],
    initialPredictionsDTO = [],
    mode = 'annotation',
    isReadOnly = false,
}: Partial<Omit<AnnotationDocumentProviderProps, 'children'>> = {}) => {
    const wrapper = ({ children }: { children: ReactNode }) => (
        <AnnotationDocumentProvider
            mode={mode}
            isReadOnly={isReadOnly}
            initialAnnotationsDTO={initialAnnotationsDTO}
            initialPredictionsDTO={initialPredictionsDTO}
        >
            {children}
        </AnnotationDocumentProvider>
    );

    return renderHook(
        () => ({ ...useAnnotations(), ...useAnnotationCommands(), isReadOnly: useIsAnnotatorReadOnly() }),
        { wrapper }
    );
};

const annotationDTO: AnnotationDTO = { labels: [{ id: 'label-1' }], shape: getMockedShape({ type: 'rectangle' }) };

describe('AnnotationDocumentProvider', () => {
    it('keeps the same commands while annotations change', () => {
        const { result } = renderDocument();
        const { addAnnotations, updateAnnotations, deleteAnnotations } = result.current;

        act(() => {
            result.current.addAnnotations([getMockedShape({ type: 'rectangle' })], [{ id: 'label-1' }]);
        });

        expect(result.current.annotations).toHaveLength(1);
        expect(result.current.addAnnotations).toBe(addAnnotations);
        expect(result.current.updateAnnotations).toBe(updateAnnotations);
        expect(result.current.deleteAnnotations).toBe(deleteAnnotations);
    });

    it('resets to the initial annotations, or to the given ones', () => {
        const { result } = renderDocument({ initialAnnotationsDTO: [annotationDTO] });

        act(() => result.current.deleteAnnotations(result.current.annotations.map(({ id }) => id)));
        expect(result.current.annotations).toHaveLength(0);

        act(() => result.current.resetAnnotations());
        expect(result.current.annotations).toHaveLength(1);

        act(() => result.current.resetAnnotations([]));
        expect(result.current.annotations).toHaveLength(0);
    });

    it('renders predictions and is read-only in prediction mode', () => {
        const { result } = renderDocument({
            mode: 'prediction',
            initialPredictionsDTO: [annotationDTO, annotationDTO],
        });

        expect(result.current.annotations).toHaveLength(2);
        expect(result.current.isReadOnly).toBe(true);
    });
});

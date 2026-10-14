// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import type { AnnotationDTO } from '@/api/types';
import { act } from '@testing-library/react';
import { getMockedShape } from 'mocks/mock-annotation';

import { useUndoRedo } from '../../shared/undo-redo/undo-redo-provider.component';
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
        () => ({
            ...useAnnotations(),
            ...useAnnotationCommands(),
            isReadOnly: useIsAnnotatorReadOnly(),
            undoRedo: useUndoRedo(),
        }),
        { wrapper }
    );
};

const annotationDTO: AnnotationDTO = { labels: [{ id: 'label-1' }], shape: getMockedShape({ type: 'rectangle' }) };
const emptyLabelAnnotationDTO: AnnotationDTO = { labels: [{ id: 'empty-label' }], shape: { type: 'full_image' } };

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

    it('replaces a full-image annotation when adding a shape, in a single undo step', () => {
        const { result } = renderDocument({ initialAnnotationsDTO: [emptyLabelAnnotationDTO] });

        act(() => {
            result.current.addAnnotations([getMockedShape({ type: 'rectangle' })], [{ id: 'label-1' }]);
        });

        expect(result.current.annotations).toHaveLength(1);
        expect(result.current.annotations[0].shape.type).toBe('rectangle');

        act(() => result.current.undoRedo.undo());

        expect(result.current.annotations).toHaveLength(1);
        expect(result.current.annotations[0].shape.type).toBe('full_image');
        expect(result.current.undoRedo.canUndo).toBe(false);
    });

    it('replaces all annotations with the empty label in a single undo step', () => {
        const { result } = renderDocument({ initialAnnotationsDTO: [annotationDTO, annotationDTO] });

        act(() => result.current.addAnnotationWithEmptyLabel({ id: 'empty-label', name: 'No object', color: '#fff' }));

        expect(result.current.annotations).toHaveLength(1);
        expect(result.current.annotations[0].labels).toEqual([{ id: 'empty-label' }]);

        act(() => result.current.undoRedo.undo());

        expect(result.current.annotations).toHaveLength(2);
        expect(result.current.undoRedo.canUndo).toBe(false);
    });
});

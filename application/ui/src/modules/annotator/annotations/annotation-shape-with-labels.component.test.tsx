// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedAnnotationLabel, getMockedAnnotationLabelRef } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { EMPTY_LABEL_ID } from '../../../shared/labels';
import * as actualLabels from '../../../shared/labels' with { rstest: 'importActual' };
import type { AnnotationLabel, AnnotationLabelRef } from '../../../shared/types';
import { AnnotationShapeWithLabels } from './annotation-shape-with-labels.component';

const mockDeleteAnnotations = rs.fn();
const mockUpdateAnnotations = rs.fn();
const mockSetSelectedLabelId = rs.fn();

rs.mock('../annotation-visibility-provider.component', () => ({
    useAnnotationVisibility: () => ({
        isVisible: true,
    }),
}));

rs.mock('../annotation-document-provider.component', () => ({
    useAnnotationCommands: () => ({
        updateAnnotations: mockUpdateAnnotations,
        deleteAnnotations: mockDeleteAnnotations,
    }),
    useIsAnnotatorReadOnly: () => false,
}));

rs.mock('../annotator-labels-provider.component', () => ({
    useAnnotatorLabels: () => ({
        labels: [],
        selectedLabel: null,
        selectedLabelId: 'empty-label',
        setSelectedLabelId: mockSetSelectedLabelId,
    }),
}));

rs.mock('../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: () => ({
        mediaItem: {
            width: 100,
            height: 100,
        },
    }),
}));

rs.mock('../../../shared/labels', () => {
    return {
        ...actualLabels,
        useLabelResolver: () => ({
            getLabel: () => undefined,
            resolveAnnotationLabel: (ref: AnnotationLabelRef): AnnotationLabel | undefined => {
                const resolved = getMockedAnnotationLabel({ id: ref.id });
                return resolved;
            },
        }),
    };
});

describe('AnnotationShapeWithLabels', () => {
    beforeEach(() => {
        const project = getMockedProject({});

        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json({
                    ...project,
                    task: {
                        ...project.task,
                        task_type: 'detection',
                    },
                });
            })
        );

        rs.clearAllMocks();
    });

    it('updates a full-image annotation to have no labels when removing its last label', async () => {
        const annotation = getMockedAnnotation({
            id: 'full-image-annotation',
            shape: { type: 'full_image' },
            labels: [getMockedAnnotationLabelRef({ id: EMPTY_LABEL_ID })],
        });

        render(<AnnotationShapeWithLabels annotation={annotation} />);
        fireEvent.pointerDown(await screen.findByRole('button', { name: 'Remove label-1' }));

        expect(mockUpdateAnnotations).toHaveBeenCalledWith([{ ...annotation, labels: [] }]);
        expect(mockDeleteAnnotations).not.toHaveBeenCalled();
        expect(mockSetSelectedLabelId).toHaveBeenCalledWith(null);
    });

    it('updates a non-full-image annotation to have no labels when removing its last label', async () => {
        const annotation = getMockedAnnotation({
            id: 'rect-annotation',
            labels: [getMockedAnnotationLabelRef({ id: 'label-1' })],
        });

        render(<AnnotationShapeWithLabels annotation={annotation} />);
        fireEvent.pointerDown(await screen.findByRole('button', { name: 'Remove label-1' }));

        expect(mockUpdateAnnotations).toHaveBeenCalledWith([{ ...annotation, labels: [] }]);
        expect(mockDeleteAnnotations).not.toHaveBeenCalled();
    });
});

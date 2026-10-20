// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ZoomProvider } from '@/components/zoom/zoom.provider';
import { fireEvent, screen, waitForElementToBeRemoved } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { Annotation } from '../../../shared/types';
import * as actualDoc from '../annotation-document-provider.component' with { rstest: 'importActual' };
import { AnnotationVisibilityProvider, useAnnotationVisibility } from '../annotation-visibility-provider.component';
import * as actualVis from '../annotation-visibility-provider.component' with { rstest: 'importActual' };
import { AnnotatorLabelsProvider } from '../annotator-labels-provider.component';
import { AnnotatorProvider } from '../annotator-provider.component';
import * as actualAnnotator from '../annotator-provider.component' with { rstest: 'importActual' };
import { useSelectedAnnotations } from '../select-annotation-provider.component';
import * as actualSelect from '../select-annotation-provider.component' with { rstest: 'importActual' };
import { CanvasSettingsProvider } from '../shell/primary-toolbar/settings/canvas-settings-provider.component';
import { ToolProvider } from '../tool-provider.component';
import { AnnotationContext } from './annotation-context';
import { EditableAnnotation } from './editable-annotation.component';

const CHILD_TEST_ID = 'child-content';
const Child = () => <g data-testid={CHILD_TEST_ID} />;

const mockROI = { x: 0, y: 0, width: 1000, height: 1000 };

const canvas = rs.hoisted(() => document.createElement('div'));

rs.mock('../annotator-provider.component', () => ({
    ...actualAnnotator,
    useAnnotator: () => ({ canvasRef: { current: canvas } }),
}));

rs.mock('../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: () => ({ roi: mockROI, image: { width: mockROI.width, height: mockROI.height } }),
}));

rs.mock('../annotation-document-provider.component', () => {
    return {
        ...actualDoc,
        useAnnotationCommands: () => ({
            updateAnnotations: rs.fn(),
            deleteAnnotations: rs.fn(),
        }),
        useIsAnnotatorReadOnly: () => false,
    };
});

rs.mock('../select-annotation-provider.component', () => {
    return {
        ...actualSelect,
        useSelectedAnnotations: rs.fn(),
    };
});

rs.mock('../annotation-visibility-provider.component', () => {
    return {
        ...actualVis,
        useAnnotationVisibility: rs.fn(),
    };
});

type AnnotationOptions = {
    isVisible?: boolean;
    selectedAnnotations?: Set<string>;
};

const renderWithAnnotation = async (
    annotation: Annotation,
    { isVisible = true, selectedAnnotations = new Set<string>() }: AnnotationOptions = {}
) => {
    rs.mocked(useAnnotationVisibility).mockReturnValue({
        isVisible,
        isFocussed: false,
        toggleVisibility: rs.fn(),
        toggleFocus: rs.fn(),
    });
    const setSelectedAnnotations = rs.fn();
    rs.mocked(useSelectedAnnotations).mockReturnValue({ selectedAnnotations, setSelectedAnnotations });

    render(
        <svg>
            <ZoomProvider>
                <AnnotatorProvider>
                    <ToolProvider>
                        <AnnotationVisibilityProvider>
                            <CanvasSettingsProvider>
                                <AnnotatorLabelsProvider>
                                    <AnnotationContext.Provider value={annotation}>
                                        <EditableAnnotation>
                                            <Child />
                                        </EditableAnnotation>
                                    </AnnotationContext.Provider>
                                </AnnotatorLabelsProvider>
                            </CanvasSettingsProvider>
                        </AnnotationVisibilityProvider>
                    </ToolProvider>
                </AnnotatorProvider>
            </ZoomProvider>
        </svg>
    );

    await waitForElementToBeRemoved(screen.getByRole('progressbar'));

    return { setSelectedAnnotations };
};

describe('EditableAnnotation', () => {
    const rectangleAnnotation = getMockedAnnotation({
        id: 'rect-1',
        shape: { type: 'rectangle', x: 0, y: 0, width: 100, height: 50 },
    });
    const polygonAnnotation = getMockedAnnotation({
        id: 'poly-1',
        shape: {
            type: 'polygon',
            points: [
                { x: 0, y: 0 },
                { x: 50, y: 0 },
                { x: 25, y: 50 },
            ],
        },
    });

    beforeEach(() => {
        server.use(http.get('/api/projects/{project_id}', () => HttpResponse.json(getMockedProject({}))));
    });

    it('renders children when the annotation is not selected', async () => {
        await renderWithAnnotation(rectangleAnnotation);

        expect(screen.getByTestId(CHILD_TEST_ID)).toBeInTheDocument();
        expect(screen.queryByLabelText(`Edit bounding box points ${rectangleAnnotation.id}`)).not.toBeInTheDocument();
    });

    it('renders EditBoundingBox anchors when a rectangle annotation is selected and visible', async () => {
        await renderWithAnnotation(rectangleAnnotation, { selectedAnnotations: new Set([rectangleAnnotation.id]) });

        expect(screen.getByLabelText(`Edit bounding box points ${rectangleAnnotation.id}`)).toBeInTheDocument();
        expect(screen.queryByTestId(CHILD_TEST_ID)).not.toBeInTheDocument();
    });

    it('renders EditPolygon when a polygon annotation is selected and visible', async () => {
        await renderWithAnnotation(polygonAnnotation, { selectedAnnotations: new Set([polygonAnnotation.id]) });

        // EditPolygon renders an svg with this id
        expect(document.getElementById(`translate-polygon-${polygonAnnotation.id}`)).toBeInTheDocument();
        expect(screen.queryByTestId(CHILD_TEST_ID)).not.toBeInTheDocument();
    });

    it('renders children instead of edit anchors when annotations are hidden, even if selected', async () => {
        await renderWithAnnotation(rectangleAnnotation, {
            isVisible: false,
            selectedAnnotations: new Set([rectangleAnnotation.id]),
        });

        expect(screen.getByTestId(CHILD_TEST_ID)).toBeInTheDocument();
        expect(screen.queryByLabelText(`Edit bounding box points ${rectangleAnnotation.id}`)).not.toBeInTheDocument();
    });

    it('renders children when multiple annotations are selected', async () => {
        await renderWithAnnotation(rectangleAnnotation, {
            selectedAnnotations: new Set([rectangleAnnotation.id, 'other-annotation']),
        });

        expect(screen.getByTestId(CHILD_TEST_ID)).toBeInTheDocument();
        expect(screen.queryByLabelText(`Edit bounding box points ${rectangleAnnotation.id}`)).not.toBeInTheDocument();
    });

    it('deselects the edited annotation when left-pressing outside of it, but not when right-pressing', async () => {
        const { setSelectedAnnotations } = await renderWithAnnotation(polygonAnnotation, {
            selectedAnnotations: new Set([polygonAnnotation.id]),
        });
        // jsdom has no PointerEvent, a MouseEvent carries the pressed button
        const pointerDown = (button: number) => fireEvent(canvas, new MouseEvent('pointerdown', { button }));

        pointerDown(2);
        expect(setSelectedAnnotations).not.toHaveBeenCalled();

        pointerDown(0);
        expect(setSelectedAnnotations).toHaveBeenCalledWith(new Set());
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ZoomProvider } from '@/components/zoom/zoom.provider';
import { screen, waitForElementToBeRemoved } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import {
    AnnotationVisibilityProvider,
    useAnnotationVisibility,
} from '../../../shared/annotator/annotation-visibility-provider.component';
import { AnnotatorProvider } from '../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { ToolProvider } from '../../../shared/annotator/tool-provider.component';
import { Annotation } from '../../../shared/types';
import { CanvasSettingsProvider } from '../../dataset/media-preview/primary-toolbar/settings/canvas-settings-provider.component';
import { AnnotatorLabelsProvider } from '../annotator-labels-provider.component';
import { AnnotationContext } from './annotation-context';
import { EditableAnnotation } from './editable-annotation.component';

const CHILD_TEST_ID = 'child-content';
const Child = () => <g data-testid={CHILD_TEST_ID} />;

const mockROI = { x: 0, y: 0, width: 1000, height: 1000 };

vi.mock('../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: () => ({ roi: mockROI, image: { width: mockROI.width, height: mockROI.height } }),
}));

vi.mock('../../../shared/annotator/annotation-actions-provider.component', async (importActual) => {
    const actual =
        await importActual<typeof import('../../../shared/annotator/annotation-actions-provider.component')>();

    return {
        ...actual,
        useAnnotationActions: () => ({
            updateAnnotations: vi.fn(),
            deleteAnnotations: vi.fn(),
            isReadOnlyMode: false,
        }),
    };
});

vi.mock('../../../shared/annotator/select-annotation-provider.component', async (importActual) => {
    const actual =
        await importActual<typeof import('../../../shared/annotator/select-annotation-provider.component')>();
    return {
        ...actual,
        useSelectedAnnotations: vi.fn(),
    };
});

vi.mock('../../../shared/annotator/annotation-visibility-provider.component', async (importActual) => {
    const actual =
        await importActual<typeof import('../../../shared/annotator/annotation-visibility-provider.component')>();
    return {
        ...actual,
        useAnnotationVisibility: vi.fn(),
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
    vi.mocked(useAnnotationVisibility).mockReturnValue({
        isVisible,
        isFocussed: false,
        toggleVisibility: vi.fn(),
        toggleFocus: vi.fn(),
    });
    vi.mocked(useSelectedAnnotations).mockReturnValue({
        selectedAnnotations,
        setSelectedAnnotations: vi.fn(),
    });

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
});

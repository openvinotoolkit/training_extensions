// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMockedAnnotation } from 'mocks/mock-annotation';
import { getMockedAnnotationLabel } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import type { Annotation, AnnotationLabel, AnnotationLabelRef } from '../../../shared/types';
import {
    useAnnotationCommands,
    useAnnotations,
    useIsAnnotatorReadOnly,
} from '../annotation-actions-provider.component';
import { useAnnotationVisibility } from '../annotation-visibility-provider.component';
import { useCanvasSettings } from '../shell/primary-toolbar/settings/canvas-settings-provider.component';
import { ReadOnlyAnnotations } from './read-only-annotations.component';

vi.mock('../annotation-actions-provider.component', () => ({
    useAnnotations: vi.fn(),
    useAnnotationCommands: vi.fn(),
    useIsAnnotatorReadOnly: vi.fn(),
}));

vi.mock('../annotation-visibility-provider.component', () => ({
    useAnnotationVisibility: vi.fn(),
}));

vi.mock('../shell/primary-toolbar/settings/canvas-settings-provider.component', () => ({
    useCanvasSettings: vi.fn(),
}));

vi.mock('../../../shared/labels', async (importOriginal) => {
    const actual = await importOriginal<typeof import('../../../shared/labels')>();
    return {
        ...actual,
        useLabelResolver: () => ({
            getLabel: () => undefined,
            resolveAnnotationLabel: (ref: AnnotationLabelRef): AnnotationLabel | undefined => {
                return getMockedAnnotationLabel({ id: ref.id, name: ref.id });
            },
        }),
    };
});

vi.mock('../annotator-labels-provider.component', () => ({
    useAnnotatorLabels: () => ({
        labels: [],
        selectedLabel: null,
        selectedLabelId: null,
        setSelectedLabelId: vi.fn(),
    }),
}));

vi.mock('../selected-media-item-provider.component', () => ({
    useSelectedMediaItem: () => ({
        mediaItem: { width: 800, height: 600 },
        roi: { x: 0, y: 0, width: 800, height: 600 },
        image: { width: 800, height: 600 },
    }),
}));

vi.mock('../../../components/zoom/zoom.provider', () => ({
    useZoom: () => ({ scale: 1 }),
}));

const defaultAnnotation = getMockedAnnotation();

type SetupOptions = {
    annotations?: Annotation[];
    isFocussed?: boolean;
    isVisible?: boolean;
    isReadOnlyMode?: boolean;
    hideLabels?: boolean;
};

const setupMocks = ({
    annotations = [defaultAnnotation],
    isFocussed = false,
    isVisible = true,
    isReadOnlyMode = true,
    hideLabels = false,
}: SetupOptions = {}) => {
    vi.mocked(useAnnotations).mockReturnValue({ annotations, initialAnnotations: [], initialPredictions: [] });
    vi.mocked(useIsAnnotatorReadOnly).mockReturnValue(isReadOnlyMode);
    vi.mocked(useAnnotationCommands).mockReturnValue({
        updateAnnotations: vi.fn(),
        deleteAnnotations: vi.fn(),
    } as unknown as ReturnType<typeof useAnnotationCommands>);

    vi.mocked(useAnnotationVisibility).mockReturnValue({
        isFocussed,
        isVisible,
        toggleVisibility: vi.fn(),
        toggleFocus: vi.fn(),
    });

    vi.mocked(useCanvasSettings).mockReturnValue({
        canvasSettings: {
            hideLabels: { value: hideLabels, defaultValue: false },
            annotationFillOpacity: { value: 1, defaultValue: 1 },
            annotationBorderOpacity: { value: 1, defaultValue: 1 },
            imageBrightness: { value: 0, defaultValue: 0 },
            imageContrast: { value: 0, defaultValue: 0 },
            imageSaturation: { value: 0, defaultValue: 0 },
            pixelView: { value: false, defaultValue: false },
        },
        setCanvasSettings: vi.fn(),
    });
};

describe('ReadOnlyAnnotations', () => {
    beforeEach(() => {
        setupMocks();
    });

    it('renders the annotations SVG element', () => {
        render(<ReadOnlyAnnotations width={800} height={600} />);

        expect(screen.getByLabelText('annotations')).toBeInTheDocument();
    });

    it('renders a shape for each annotation', () => {
        setupMocks({
            annotations: [
                getMockedAnnotation({
                    id: 'ann-1',
                    shape: { type: 'rectangle', x: 10, y: 20, width: 100, height: 50 },
                }),
            ],
        });

        render(<ReadOnlyAnnotations width={800} height={600} />);

        expect(screen.getAllByLabelText('annotation rect')).toHaveLength(1);
    });

    it('renders each annotation twice while the focus mask is active', () => {
        setupMocks({
            isFocussed: true,
            annotations: [
                getMockedAnnotation({
                    id: 'ann-1',
                    shape: { type: 'rectangle', x: 10, y: 20, width: 100, height: 50 },
                }),
            ],
        });

        render(<ReadOnlyAnnotations width={800} height={600} />);

        expect(screen.getAllByLabelText('annotation rect')).toHaveLength(2); // 1 in mask, 1 visible
    });

    it('renders no shapes when there are no annotations', () => {
        setupMocks({ annotations: [] });

        render(<ReadOnlyAnnotations width={800} height={600} />);

        expect(screen.queryByLabelText('annotation rect')).not.toBeInTheDocument();
    });

    it('activates the focus mask when isFocussed is true', () => {
        setupMocks({ isFocussed: true });

        const { container } = render(<ReadOnlyAnnotations width={800} height={600} />);

        const maskOverlay = container.querySelector('rect[mask]') as SVGRectElement | null;

        expect(maskOverlay).toBeInTheDocument();
        expect(maskOverlay?.style.fillOpacity).toBe('0.3');
    });

    it('omits the focus mask entirely when isFocussed is false', () => {
        setupMocks({ isFocussed: false });

        const { container } = render(<ReadOnlyAnnotations width={800} height={600} />);

        const maskOverlay = container.querySelector('rect[mask]') as SVGRectElement | null;

        expect(maskOverlay).not.toBeInTheDocument();
    });

    it('renders annotation labels when hideLabels is false', () => {
        setupMocks({ hideLabels: false });

        render(<ReadOnlyAnnotations width={800} height={600} />);

        expect(screen.getByText('label-1')).toBeInTheDocument();
    });

    // Visual hiding of labels is applied via the `--annotation-labels-display` CSS variable set by
    // AnnotatorCanvasSettings (an ancestor outside this component), not by conditionally mounting
    // them here — see annotator-canvas-settings.component.test.tsx. Labels stay in the DOM either way.
    it('keeps annotation labels mounted when hideLabels is true (hidden via CSS, not unmounted)', () => {
        setupMocks({ hideLabels: true });

        render(<ReadOnlyAnnotations width={800} height={600} />);

        expect(screen.getByText('label-1')).toBeInTheDocument();
    });
});

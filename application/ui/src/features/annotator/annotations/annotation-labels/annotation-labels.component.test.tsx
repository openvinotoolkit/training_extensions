// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedAnnotationLabel, getMockedAnnotationLabelRef } from 'mocks/mock-labels';
import { render } from 'test-utils/render';

import type { AnnotationLabel, AnnotationLabelRef } from '../../../../shared/types';
import { AnnotationLabels } from './annotation-labels.component';

// Resolve refs by returning a mock AnnotationLabel matching the ref id
vi.mock('../../../../shared/annotator/labels', async (importOriginal) => {
    const actual = await importOriginal<typeof import('../../../../shared/annotator/labels')>();
    return {
        ...actual,
        useLabelResolver: () => ({
            getLabel: () => undefined,
            resolveAnnotationLabel: (ref: AnnotationLabelRef): AnnotationLabel | undefined => {
                return getMockedAnnotationLabel({ id: ref.id, name: ref.id, color: '#FF0000' });
            },
        }),
    };
});

describe('AnnotationLabels', () => {
    const mockOnRemove = vi.fn();

    afterEach(() => {
        mockOnRemove.mockClear();
    });

    it('renders placeholder when no labels provided', () => {
        render(
            <svg>
                <AnnotationLabels labels={[]} onRemove={mockOnRemove} />
            </svg>
        );

        expect(screen.getByText('No label')).toBeInTheDocument();
    });

    it('renders single label with name and color resolved from catalog', () => {
        const labelRef = getMockedAnnotationLabelRef({ id: 'label-1' });

        render(
            <svg>
                <AnnotationLabels labels={[labelRef]} onRemove={mockOnRemove} />
            </svg>
        );

        expect(screen.getByText('label-1')).toBeInTheDocument();

        const labelElement = screen.getByLabelText('label label-1 background');
        expect(labelElement).toHaveStyle({ '--label-color': '#FF0000' });
    });

    it('renders multiple labels horizontally', () => {
        const refs: AnnotationLabelRef[] = [
            getMockedAnnotationLabelRef({ id: 'label-1' }),
            getMockedAnnotationLabelRef({ id: 'label-2' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={refs} onRemove={mockOnRemove} />
            </svg>
        );

        expect(screen.getAllByText('label-1').length).toBeGreaterThan(0);
        expect(screen.getAllByText('label-2').length).toBeGreaterThan(0);
    });

    it('calls onRemove when close button clicked', () => {
        const labelRef = getMockedAnnotationLabelRef({ id: 'label-1' });

        render(
            <svg>
                <AnnotationLabels labels={[labelRef]} onRemove={mockOnRemove} />
            </svg>
        );

        const closeButton = screen.getByLabelText('Remove label-1');
        fireEvent.pointerDown(closeButton);

        expect(mockOnRemove).toHaveBeenCalledTimes(1);
        expect(mockOnRemove).toHaveBeenCalledWith('label-1');
    });

    it('does not render remove button when labels are non-removable', () => {
        const labelRef = getMockedAnnotationLabelRef({ id: 'label-1' });

        render(
            <svg>
                <AnnotationLabels labels={[labelRef]} onRemove={mockOnRemove} isRemovable={false} />
            </svg>
        );

        expect(screen.queryByLabelText('Remove label-1')).not.toBeInTheDocument();
    });

    it('positions foreignObject just above annotation anchor for CSS scaling', () => {
        const labelRef = getMockedAnnotationLabelRef({ id: 'label-1' });

        render(
            <svg>
                <AnnotationLabels labels={[labelRef]} onRemove={mockOnRemove} />
            </svg>
        );

        const foreignObject = document.querySelector('foreignObject');
        expect(foreignObject).toHaveAttribute('height', '24');
        expect(foreignObject).toHaveAttribute('y', '-24');
    });

    it('prevents event propagation on close button click', () => {
        const labelRef = getMockedAnnotationLabelRef({ id: 'label-1' });
        const mockParentHandler = vi.fn();

        render(
            <svg onPointerDown={mockParentHandler}>
                <AnnotationLabels labels={[labelRef]} onRemove={mockOnRemove} />
            </svg>
        );

        const closeButton = screen.getByLabelText('Remove label-1');
        fireEvent.pointerDown(closeButton);

        expect(mockOnRemove).toHaveBeenCalled();
        // Parent handler should not be called due to stopPropagation
        expect(mockParentHandler).not.toHaveBeenCalled();
    });

    it('renders labels in correct order', () => {
        const labelRefs: AnnotationLabelRef[] = [
            getMockedAnnotationLabelRef({ id: 'label-1' }),
            getMockedAnnotationLabelRef({ id: 'label-2' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={labelRefs} onRemove={mockOnRemove} />
            </svg>
        );

        const firstLabel = screen.getByLabelText('label label-1 background');
        const secondLabel = screen.getByLabelText('label label-2 background');

        expect(firstLabel).toBeInTheDocument();
        expect(secondLabel).toBeInTheDocument();

        expect(firstLabel.parentElement).toBe(secondLabel.parentElement);
    });

    it('applies correct border radius for first and last labels', () => {
        const refs: AnnotationLabelRef[] = [
            getMockedAnnotationLabelRef({ id: 'label-1' }),
            getMockedAnnotationLabelRef({ id: 'label-2' }),
            getMockedAnnotationLabelRef({ id: 'label-3' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={refs} onRemove={mockOnRemove} />
            </svg>
        );

        const firstLabel = screen.getByLabelText('label label-1 background');
        const middleLabel = screen.getByLabelText('label label-2 background');
        const lastLabel = screen.getByLabelText('label label-3 background');

        // First label should have top-left rounded
        expect(firstLabel).toHaveStyle({
            '--border-top-left': 'var(--spectrum-global-dimension-size-50)',
            '--border-top-right': '0',
        });
        // Middle label should have no rounded corners
        expect(middleLabel).toHaveStyle({ '--border-top-left': '0', '--border-top-right': '0' });
        // Last label should have top-right rounded
        expect(lastLabel).toHaveStyle({
            '--border-top-left': '0',
            '--border-top-right': 'var(--spectrum-global-dimension-size-50)',
        });
    });

    it('applies bottom corners when useBottomCorners is true', () => {
        const labelRefs: AnnotationLabelRef[] = [
            getMockedAnnotationLabelRef({ id: 'label-1' }),
            getMockedAnnotationLabelRef({ id: 'label-2' }),
        ];

        render(
            <svg>
                <AnnotationLabels labels={labelRefs} onRemove={mockOnRemove} useBottomCorners />
            </svg>
        );

        const firstLabel = screen.getByLabelText('label label-1 background');
        const lastLabel = screen.getByLabelText('label label-2 background');

        // First label should have bottom-left rounded
        expect(firstLabel).toHaveStyle({
            '--border-bottom-left': 'var(--spectrum-global-dimension-size-50)',
            '--border-bottom-right': '0',
        });
        // Last label should have bottom-right rounded
        expect(lastLabel).toHaveStyle({
            '--border-bottom-left': '0',
            '--border-bottom-right': 'var(--spectrum-global-dimension-size-50)',
        });
    });
});

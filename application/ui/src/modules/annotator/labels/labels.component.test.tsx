// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '@/api/types';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { EMPTY_LABEL_ID } from '../../../shared/labels';
import type { AnnotationLabelRef } from '../../../shared/types';
import { Labels } from './labels.component';

const mockLabels: Label[] = [
    getMockedLabel({ id: 'label-1', name: 'Person', color: '#FF0000', hotkey: 'p' }),
    getMockedLabel({ id: 'label-2', name: 'Car', color: '#00FF00' }),
    getMockedLabel({ id: 'label-3', name: 'Dog', color: '#0000FF' }),
    getMockedLabel({ id: EMPTY_LABEL_ID, name: 'No object', color: '#FFFFFF' }),
];

const mockSelectedLabelId = { current: 'label-1' };

const mockSetSelectedLabelId = vi.fn();
const mockUpdateAnnotations = vi.fn();
const mockAddAnnotations = vi.fn();
const mockDeleteAnnotations = vi.fn();
const mockSelectedAnnotations = { current: new Set<string>() };
const mockAnnotations = { current: [] as { id: string; labels: AnnotationLabelRef[]; shape: { type: string } }[] };

vi.mock('../annotator-labels-provider.component', () => ({
    useAnnotatorLabels: () => ({
        labels: mockLabels,
        selectedLabelId: mockSelectedLabelId.current,
        setSelectedLabelId: mockSetSelectedLabelId,
    }),
}));

vi.mock('../select-annotation-provider.component', () => ({
    useSelectedAnnotations: () => ({
        selectedAnnotations: mockSelectedAnnotations.current,
    }),
}));

vi.mock('../annotation-actions-provider.component', () => ({
    useAnnotations: () => ({ annotations: mockAnnotations.current }),
    useAnnotationCommands: () => ({
        updateAnnotations: mockUpdateAnnotations,
        addAnnotations: mockAddAnnotations,
        deleteAnnotations: mockDeleteAnnotations,
        addAnnotationWithEmptyLabel: vi.fn(),
    }),
}));

vi.mock('./api/use-update-label.hook', () => ({
    useUpdateLabel: () => ({
        mutate: vi.fn(),
    }),
}));

describe('Labels', () => {
    beforeEach(() => {
        mockSetSelectedLabelId.mockClear();
        mockUpdateAnnotations.mockClear();
        mockAddAnnotations.mockClear();
        mockDeleteAnnotations.mockClear();
        mockSelectedLabelId.current = 'label-1';
        mockSelectedAnnotations.current = new Set();
        mockAnnotations.current = [];

        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject());
            })
        );
    });

    it('renders all labels as badges', async () => {
        render(<Labels />);

        expect(await screen.findByRole('button', { name: 'Label Person' })).toBeInTheDocument();
        expect(await screen.findByRole('button', { name: 'Label Car' })).toBeInTheDocument();
        expect(await screen.findByRole('button', { name: 'Label Dog' })).toBeInTheDocument();
    });

    it('shows selected label with aria-pressed true', async () => {
        render(<Labels />);

        const personButton = await screen.findByRole('button', { name: 'Label Person' });
        const carButton = await screen.findByRole('button', { name: 'Label Car' });

        expect(personButton).toHaveAttribute('aria-pressed', 'true');
        expect(carButton).toHaveAttribute('aria-pressed', 'false');
    });

    it('calls setSelectedLabelId when clicking a label', async () => {
        const user = userEvent.setup();
        render(<Labels />);

        const carButton = await screen.findByRole('button', { name: 'Label Car' });
        await user.click(carButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-2');
    });

    it('displays label names in badges', async () => {
        render(<Labels />);

        expect(await screen.findByText('Person')).toBeInTheDocument();
        expect(await screen.findByText('Car')).toBeInTheDocument();
        expect(await screen.findByText('Dog')).toBeInTheDocument();
    });

    it('selects label when pressing configured hotkey', async () => {
        render(<Labels />);

        await screen.findByRole('button', { name: 'Label Person' });
        fireEvent.keyDown(document, { key: 'p', code: 'KeyP' });

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-1');
    });

    it('updates selected annotations when clicking a different label', async () => {
        const user = userEvent.setup();
        mockSelectedAnnotations.current = new Set(['annotation-1']);
        mockAnnotations.current = [
            { id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [{ id: 'label-2' }], shape: { type: 'RECTANGLE' } },
        ];

        render(<Labels />);

        const carButton = await screen.findByRole('button', { name: 'Label Car' });
        await user.click(carButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-2');
        expect(mockUpdateAnnotations).toHaveBeenCalledWith([mockAnnotations.current[0]], [{ id: 'label-2' }]);
    });

    it('does not update annotations when no annotations are selected', async () => {
        const user = userEvent.setup();
        mockSelectedAnnotations.current = new Set();
        mockAnnotations.current = [{ id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'RECTANGLE' } }];

        render(<Labels />);

        const carButton = await screen.findByRole('button', { name: 'Label Car' });
        await user.click(carButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-2');
        expect(mockUpdateAnnotations).not.toHaveBeenCalled();
    });

    it('removes label when all selected annotations already have it', async () => {
        const user = userEvent.setup();
        mockSelectedAnnotations.current = new Set(['annotation-1', 'annotation-2']);
        mockAnnotations.current = [
            { id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [{ id: 'label-1' }], shape: { type: 'RECTANGLE' } },
        ];

        render(<Labels />);

        const personButton = await screen.findByRole('button', { name: 'Label Person' });
        await user.click(personButton);

        expect(mockUpdateAnnotations).toHaveBeenCalledWith([
            { id: 'annotation-1', labels: [], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [], shape: { type: 'RECTANGLE' } },
        ]);
        expect(mockSetSelectedLabelId).toHaveBeenCalledWith(null);
    });

    it('adds label when at least one selected annotation does not have it', async () => {
        const user = userEvent.setup();
        mockSelectedAnnotations.current = new Set(['annotation-1', 'annotation-2']);
        mockAnnotations.current = [
            { id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'RECTANGLE' } },
            { id: 'annotation-2', labels: [{ id: 'label-2' }], shape: { type: 'RECTANGLE' } },
        ];

        render(<Labels />);

        const personButton = await screen.findByRole('button', { name: 'Label Person' });
        await user.click(personButton);

        expect(mockSetSelectedLabelId).toHaveBeenCalledWith('label-1');
        expect(mockUpdateAnnotations).toHaveBeenCalledWith(
            [mockAnnotations.current[0], mockAnnotations.current[1]],
            [{ id: 'label-1' }]
        );
    });

    describe('classification mode', () => {
        it('creates full_image annotation when no annotations exist', async () => {
            const user = userEvent.setup();
            mockAnnotations.current = [];

            render(<Labels isClassification={true} />);

            const personButton = await screen.findByRole('button', { name: 'Label Person' });
            await user.click(personButton);

            expect(mockAddAnnotations).toHaveBeenCalledWith([{ type: 'full_image' }], [{ id: 'label-1' }]);
        });

        it('replaces labels in single-label mode', async () => {
            const user = userEvent.setup();
            mockAnnotations.current = [
                { id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'full_image' } },
            ];

            render(<Labels isClassification={true} isMultiLabel={false} />);

            const carButton = await screen.findByRole('button', { name: 'Label Car' });
            await user.click(carButton);

            expect(mockUpdateAnnotations).toHaveBeenCalledWith([
                { ...mockAnnotations.current[0], labels: [{ id: 'label-2' }] },
            ]);
        });

        it('toggles label on in multi-label mode', async () => {
            const user = userEvent.setup();
            mockAnnotations.current = [
                { id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'full_image' } },
            ];

            render(<Labels isClassification={true} isMultiLabel={true} />);

            const carButton = await screen.findByRole('button', { name: 'Label Car' });
            await user.click(carButton);

            expect(mockUpdateAnnotations).toHaveBeenCalledWith([
                { ...mockAnnotations.current[0], labels: [{ id: 'label-1' }, { id: 'label-2' }] },
            ]);
        });

        it('toggles label off in multi-label mode', async () => {
            const user = userEvent.setup();
            mockAnnotations.current = [
                { id: 'annotation-1', labels: [{ id: 'label-1' }, { id: 'label-2' }], shape: { type: 'full_image' } },
            ];

            render(<Labels isClassification={true} isMultiLabel={true} />);

            const personButton = await screen.findByRole('button', { name: 'Label Person' });
            await user.click(personButton);

            expect(mockUpdateAnnotations).toHaveBeenCalledWith([
                { ...mockAnnotations.current[0], labels: [{ id: 'label-2' }] },
            ]);
        });

        it('keeps annotation with empty labels when removing last label in multi-label mode', async () => {
            const user = userEvent.setup();
            mockAnnotations.current = [
                { id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'full_image' } },
            ];

            render(<Labels isClassification={true} isMultiLabel={true} />);

            const personButton = await screen.findByRole('button', { name: 'Label Person' });
            await user.click(personButton);

            expect(mockUpdateAnnotations).toHaveBeenCalledWith([{ ...mockAnnotations.current[0], labels: [] }]);
            expect(mockDeleteAnnotations).not.toHaveBeenCalled();
        });

        it('does not require selected annotations for classification', async () => {
            const user = userEvent.setup();
            mockSelectedAnnotations.current = new Set(); // No annotations selected
            mockAnnotations.current = [
                { id: 'annotation-1', labels: [{ id: 'label-1' }], shape: { type: 'full_image' } },
            ];

            render(<Labels isClassification={true} />);

            const carButton = await screen.findByRole('button', { name: 'Label Car' });
            await user.click(carButton);

            expect(mockUpdateAnnotations).toHaveBeenCalled();
        });

        it('shows badge as selected when label is applied to annotation', async () => {
            mockAnnotations.current = [
                { id: 'annotation-1', labels: [{ id: 'label-1' }, { id: 'label-2' }], shape: { type: 'full_image' } },
            ];

            render(<Labels isClassification={true} isMultiLabel={true} />);

            const personButton = await screen.findByRole('button', { name: 'Label Person' });
            const carButton = await screen.findByRole('button', { name: 'Label Car' });
            const dogButton = await screen.findByRole('button', { name: 'Label Dog' });

            expect(personButton).toHaveAttribute('aria-pressed', 'true');
            expect(carButton).toHaveAttribute('aria-pressed', 'true');
            expect(dogButton).toHaveAttribute('aria-pressed', 'false');
        });
    });
});

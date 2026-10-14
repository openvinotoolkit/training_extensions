// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '@/api/types';
import { waitFor } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { useLabels } from './use-labels.hook';

const mockLabels: Label[] = [
    getMockedLabel({ id: 'label-1', name: 'Person', color: '#FF0000', hotkey: 'q' }),
    getMockedLabel({ id: 'label-2', name: 'Car', color: '#00FF00', hotkey: 'w' }),
    getMockedLabel({ id: 'label-3', name: 'Dog', color: '#0000FF' }),
];

vi.mock('../annotator-labels-provider.component', () => ({
    useAnnotatorLabels: () => ({
        labels: mockLabels,
        selectedLabelId: 'label-1',
        setSelectedLabelId: vi.fn(),
    }),
}));

vi.mock('../select-annotation-provider.component', () => ({
    useSelectedAnnotations: () => ({
        selectedAnnotations: new Set<string>(),
    }),
}));

vi.mock('../annotation-document-provider.component', () => ({
    useAnnotations: () => ({ annotations: [] }),
    useAnnotationCommands: () => ({
        updateAnnotations: vi.fn(),
        addAnnotations: vi.fn(),
        addAnnotationWithEmptyLabel: vi.fn(),
    }),
}));

describe('validateHotkey', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(
                    getMockedProject({ task: { task_type: 'detection', exclusive_labels: true, labels: [] } })
                );
            }),
            http.patch('/api/projects/{project_id}/labels', () => {
                return HttpResponse.json([]);
            })
        );
    });

    it('returns undefined for a valid hotkey', async () => {
        const { result } = renderHook(() => useLabels());

        await waitFor(() => {
            expect(result.current.validateHotkey('x')).toBeUndefined();
        });
    });

    it('returns error when hotkey conflicts with an existing label hotkey', async () => {
        const { result } = renderHook(() => useLabels());

        await waitFor(() => {
            expect(result.current.validateHotkey('q')).toBe('That hotkey is already in use');
        });
    });

    it('returns undefined when conflicting label is excluded by id', async () => {
        const { result } = renderHook(() => useLabels());

        await waitFor(() => {
            expect(result.current.validateHotkey('q', 'label-1')).toBeUndefined();
        });
    });

    it('returns error when hotkey conflicts with an app hotkey', async () => {
        const { result } = renderHook(() => useLabels());

        // 'b' is the boundingBoxTool hotkey for detection task
        await waitFor(() => {
            expect(result.current.validateHotkey('b')).toBe('That hotkey is already in use');
        });
    });

    it('returns undefined for empty hotkey', async () => {
        const { result } = renderHook(() => useLabels());

        await waitFor(() => {
            expect(result.current.validateHotkey('')).toBeUndefined();
        });
    });
});

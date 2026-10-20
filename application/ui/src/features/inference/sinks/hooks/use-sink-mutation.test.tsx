// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { LocalFolderSinkConfig } from '@/api/types';
import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { useSinkMutation } from './use-sink-mutation.hook';

const mockedSource: LocalFolderSinkConfig = {
    id: 'original-id',
    name: 'Mock Source',
    output_formats: [],
    sink_type: 'folder',
    folder_path: './folder/111',
};

describe('useSinkMutation', () => {
    it('creates a new sink and return its resource id', async () => {
        const newResourceId = 'resource-id-123';
        const testSinkSpy = rs.fn();
        const { result } = renderHook(() => useSinkMutation(true));

        server.use(
            http.post('/api/sinks', () => HttpResponse.json({ ...mockedSource, id: newResourceId })),
            http.post('/api/sinks/{sink_id}:test', () => {
                testSinkSpy();

                return HttpResponse.json({
                    reachable: true,
                    latency_ms: 5,
                    error: null,
                });
            }),
            http.patch('/api/sinks/{sink_id}', () => HttpResponse.error())
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(newResourceId);
        });

        await waitFor(() => {
            expect(testSinkSpy).toHaveBeenCalled();
        });
    });

    it('update a source item and returns its resource id', async () => {
        const testSinkSpy = rs.fn();
        const { result } = renderHook(() => useSinkMutation(false));

        server.use(
            http.post('/api/sinks', () => HttpResponse.error()),
            http.post('/api/sinks/{sink_id}:test', () => {
                testSinkSpy();

                return HttpResponse.json({
                    reachable: false,
                    latency_ms: null,
                    error: 'Unavailable',
                });
            }),
            http.patch('/api/sinks/{sink_id}', () => HttpResponse.json(mockedSource))
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(mockedSource.id);
        });

        await waitFor(() => {
            expect(testSinkSpy).toHaveBeenCalled();
        });
    });
});

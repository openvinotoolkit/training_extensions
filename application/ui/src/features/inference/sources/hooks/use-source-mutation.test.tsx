// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { USBCameraSourceConfig } from '@/api/types';
import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { useSourceMutation } from './use-source-mutation.hook';

const mockedSource: USBCameraSourceConfig = {
    id: 'original-id',
    name: 'Mock Source',
    source_type: 'usb_camera' as const,
    device_id: 0,
};

describe('useSourceMutation', () => {
    it('creates a new source and return its resource id', async () => {
        const newResourceId = 'resource-id-123';
        const testSourceSpy = rs.fn();
        const { result } = renderHook(() => useSourceMutation(true));

        server.use(
            http.post('/api/sources', () => HttpResponse.json({ ...mockedSource, id: newResourceId })),
            http.post('/api/sources/{source_id}:test', () => {
                testSourceSpy();

                return HttpResponse.json({
                    reachable: true,
                    latency_ms: 10,
                    error: null,
                });
            }),
            http.patch('/api/sources/{source_id}', () => HttpResponse.error())
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(newResourceId);
        });

        await waitFor(() => {
            expect(testSourceSpy).toHaveBeenCalled();
        });
    });

    it('update a source item and returns its resource id', async () => {
        const testSourceSpy = rs.fn();
        const { result } = renderHook(() => useSourceMutation(false));

        server.use(
            http.post('/api/sources', () => HttpResponse.error()),
            http.post('/api/sources/{source_id}:test', () => {
                testSourceSpy();

                return HttpResponse.json({
                    reachable: false,
                    latency_ms: null,
                    error: 'Unavailable',
                });
            }),
            http.patch('/api/sources/{source_id}', () => HttpResponse.json(mockedSource))
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(mockedSource.id);
        });

        await waitFor(() => {
            expect(testSourceSpy).toHaveBeenCalled();
        });
    });
});

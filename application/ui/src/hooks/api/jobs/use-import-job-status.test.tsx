// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../../mocks/mock-job';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import {
    getLastEventSource,
    MockEventSourceConstructor,
    resetMockEventSource,
    simulateSSEMessage,
} from '../../../test-utils/mock-event-source';
import { useImportJobStatus } from './use-import-job-status.hook';

describe('useImportJobStatus', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        resetMockEventSource();
    });

    it('calls onSuccess when job finishes successfully', async () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(getMockedPrepareImportDatasetJob({ job_id: params.job_id, status: 'DONE' }), {
                    status: 200,
                });
            })
        );

        const { result } = renderHook(() => useImportJobStatus({ jobId: 'job-123', onSuccess, onError }));

        await waitFor(() => {
            expect(result.current.data?.status).toBe('DONE');
        });

        expect(onSuccess).toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
    });

    it('does not call onSuccess again when its callback changes', async () => {
        const firstOnSuccess = vi.fn();
        let onSuccess = firstOnSuccess;

        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(getMockedPrepareImportDatasetJob({ job_id: params.job_id, status: 'DONE' }), {
                    status: 200,
                });
            })
        );

        const { rerender } = renderHook(() => useImportJobStatus({ jobId: 'job-callback', onSuccess }));

        await waitFor(() => {
            expect(firstOnSuccess).toHaveBeenCalledOnce();
        });

        onSuccess = vi.fn();
        rerender();

        expect(onSuccess).not.toHaveBeenCalled();
    });

    it('calls onError when job fails', async () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(
                    getMockedPrepareImportDatasetJob({ job_id: params.job_id, status: 'FAILED' }),
                    { status: 200 }
                );
            })
        );

        const { result } = renderHook(() => useImportJobStatus({ jobId: 'job-456', onSuccess, onError }));

        await waitFor(() => {
            expect(result.current.data?.status).toBe('FAILED');
        });

        expect(onError).toHaveBeenCalled();
        expect(onSuccess).not.toHaveBeenCalled();
    });

    it('handles running status without calling callbacks', async () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(
                    getMockedPrepareImportDatasetJob({
                        job_id: params.job_id,
                        status: 'RUNNING',
                        progress: 50,
                    }),
                    { status: 200 }
                );
            })
        );

        const { result } = renderHook(() => useImportJobStatus({ jobId: 'job-789', onSuccess, onError }));

        await waitFor(() => {
            expect(result.current.data?.progress).toBe(50);
        });

        expect(onSuccess).not.toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
    });

    it('does not poll when jobId is not provided', () => {
        const onError = vi.fn();
        const onSuccess = vi.fn();

        const { result } = renderHook(() => useImportJobStatus({ jobId: undefined, onSuccess, onError }));

        expect(result.current.data).toBeUndefined();
        expect(onSuccess).not.toHaveBeenCalled();
        expect(onError).not.toHaveBeenCalled();
    });

    it('does not open an SSE connection when jobId is not provided', () => {
        renderHook(() => useImportJobStatus({ jobId: undefined }));

        expect(MockEventSourceConstructor).not.toHaveBeenCalled();
    });

    it('updates the job from the SSE status stream instead of polling', async () => {
        server.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                return HttpResponse.json(
                    getMockedPrepareImportDatasetJob({ job_id: params.job_id, status: 'RUNNING', progress: 10 }),
                    { status: 200 }
                );
            })
        );

        const onSuccess = vi.fn();
        const { result } = renderHook(() => useImportJobStatus({ jobId: 'job-sse', onSuccess }));

        await waitFor(() => {
            expect(result.current.data?.progress).toBe(10);
        });

        expect(MockEventSourceConstructor).toHaveBeenCalledWith(expect.stringContaining('/api/jobs/job-sse/status'));

        const eventSource = getLastEventSource();

        act(() => {
            simulateSSEMessage(
                eventSource,
                getMockedPrepareImportDatasetJob({ job_id: 'job-sse', status: 'RUNNING', progress: 75 })
            );
        });

        await waitFor(() => {
            expect(result.current.data?.progress).toBe(75);
        });

        act(() => {
            simulateSSEMessage(eventSource, getMockedPrepareImportDatasetJob({ job_id: 'job-sse', status: 'DONE' }));
        });

        await waitFor(() => {
            expect(onSuccess).toHaveBeenCalled();
        });

        // The stream is closed once the job reaches a terminal state
        expect(eventSource.close).toHaveBeenCalled();
    });
});

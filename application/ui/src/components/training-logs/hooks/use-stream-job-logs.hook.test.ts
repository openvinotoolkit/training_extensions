// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { getMockedLogEntry } from 'mocks/mock-log-entry';
import {
    getLastEventSource,
    MockEventSourceConstructor,
    resetMockEventSource,
    simulateSSEError,
    simulateSSEMessage,
    simulateSSEOpen,
} from 'test-utils/mock-event-source';
import { renderHook } from 'test-utils/render';

import { useStreamJobLogs } from './use-stream-job-logs.hook';

describe('useStreamJobLogs', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('does not create an EventSource when jobId is undefined', () => {
        renderHook(() => useStreamJobLogs(undefined, { retry: true }));

        expect(MockEventSourceConstructor).not.toHaveBeenCalled();
    });

    it('creates an EventSource with the correct URL when jobId is provided', () => {
        renderHook(() => useStreamJobLogs('job-1', { retry: true }));

        expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1);
        expect(getLastEventSource().url).toContain('/api/jobs/job-1/logs');
    });

    it('accumulates log entries as SSE messages arrive', async () => {
        const { result } = renderHook(() => useStreamJobLogs('job-1', { retry: true }));
        const eventSource = getLastEventSource();

        act(() => {
            simulateSSEMessage(eventSource, getMockedLogEntry({ message: 'First log' }));
        });

        await waitFor(() => {
            expect(result.current.logs).toHaveLength(1);
            expect(result.current.logs[0].record.message).toBe('First log');
        });

        act(() => {
            simulateSSEMessage(eventSource, getMockedLogEntry({ message: 'Second log' }));
        });

        await waitFor(() => {
            expect(result.current.logs).toHaveLength(2);
        });
    });

    it('sets connectionStatus to connecting on SSE error (retry)', async () => {
        const { result } = renderHook(() => useStreamJobLogs('job-1', { retry: true }));
        const eventSource = getLastEventSource();

        act(() => {
            simulateSSEOpen(eventSource);
        });

        await waitFor(() => {
            expect(result.current.connectionStatus).toBe('connected');
        });

        act(() => {
            simulateSSEError(eventSource);
        });

        await waitFor(() => {
            expect(result.current.connectionStatus).toBe('connecting');
        });
    });

    it('preserves log order across multiple consecutive messages', async () => {
        const { result } = renderHook(() => useStreamJobLogs('job-1', { retry: true }));
        const eventSource = getLastEventSource();

        act(() => {
            simulateSSEMessage(eventSource, getMockedLogEntry({ message: 'Message 1' }));
            simulateSSEMessage(eventSource, getMockedLogEntry({ message: 'Message 2' }));
            simulateSSEMessage(eventSource, getMockedLogEntry({ message: 'Message 3' }));
        });

        await waitFor(() => {
            expect(result.current.logs).toHaveLength(3);
            expect(result.current.logs.map((log) => log.record.message)).toEqual([
                'Message 1',
                'Message 2',
                'Message 3',
            ]);
        });
    });
});

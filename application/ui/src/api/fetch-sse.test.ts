// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getLastEventSource,
    MockEventSourceConstructor,
    resetMockEventSource,
    simulateSSEError,
    simulateSSEMessage,
    simulateSSEOpen,
} from '../test-utils/mock-event-source';
import { connectSSE } from './fetch-sse';

describe('connectSSE', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('creates an EventSource with the correct URL including API base', () => {
        connectSSE('/api/jobs/123/status', { onMessage: rs.fn() });

        expect(MockEventSourceConstructor).toHaveBeenCalledWith(expect.stringContaining('/api/jobs/123/status'));
    });

    it('parses and delivers JSON messages via onMessage callback', () => {
        const onMessage = rs.fn();
        connectSSE<{ value: number }>('/api/test', { onMessage });
        const eventSource = getLastEventSource();

        simulateSSEMessage(eventSource, { value: 42 });

        expect(onMessage).toHaveBeenCalledWith({ value: 42 });
    });

    it('delivers multiple messages in order', () => {
        const messages: number[] = [];
        connectSSE<{ n: number }>('/api/test', {
            onMessage: (data) => messages.push(data.n),
        });
        const eventSource = getLastEventSource();

        simulateSSEMessage(eventSource, { n: 1 });
        simulateSSEMessage(eventSource, { n: 2 });
        simulateSSEMessage(eventSource, { n: 3 });
        expect(messages).toEqual([1, 2, 3]);
    });

    it('closes the connection and resolves done when the server closes the connection', async () => {
        const onMessage = rs.fn();
        const onClose = rs.fn();
        const { done } = connectSSE('/api/test', { onMessage, onClose });
        const eventSource = getLastEventSource();
        simulateSSEError(eventSource);

        await done;

        expect(onClose).toHaveBeenCalledOnce();
        expect(eventSource.close).toHaveBeenCalledOnce();
    });

    it('calls onOpen when the connection is established', () => {
        const onOpen = rs.fn();
        connectSSE('/api/test', { onMessage: rs.fn(), onOpen });
        const eventSource = getLastEventSource();

        simulateSSEOpen(eventSource);
        expect(onOpen).toHaveBeenCalledOnce();
    });

    it('calls onError and onClose when an error occurs, then resolves done', async () => {
        const onError = rs.fn();
        const onClose = rs.fn();
        const { done } = connectSSE('/api/test', { onMessage: rs.fn(), onError, onClose });
        const eventSource = getLastEventSource();

        simulateSSEError(eventSource);

        await done;

        expect(onError).toHaveBeenCalledOnce();
        expect(onClose).toHaveBeenCalledOnce();
        expect(eventSource.close).toHaveBeenCalledOnce();
    });

    it('skips unparseable messages without throwing', () => {
        const onMessage = rs.fn();
        connectSSE('/api/test', { onMessage });
        const eventSource = getLastEventSource();

        eventSource.onmessage?.({ data: 'not valid json' } as MessageEvent);
        expect(onMessage).not.toHaveBeenCalled();
    });

    it('close() is idempotent — calling multiple times only triggers onClose once', () => {
        const onClose = rs.fn();
        const { close } = connectSSE('/api/test', { onMessage: rs.fn(), onClose });
        const eventSource = getLastEventSource();

        close();
        close();
        close();

        expect(onClose).toHaveBeenCalledOnce();
        expect(eventSource.close).toHaveBeenCalledOnce();
    });

    it('manual close() closes the EventSource', async () => {
        const { close, done } = connectSSE('/api/test', { onMessage: rs.fn() });
        const eventSource = getLastEventSource();

        simulateSSEOpen(eventSource);
        close();

        expect(eventSource.close).toHaveBeenCalledOnce();
        await expect(done).resolves.toBeUndefined();
    });

    it('returns the underlying EventSource instance', () => {
        const { eventSource } = connectSSE('/api/test', { onMessage: rs.fn() });
        const lastEventSource = getLastEventSource();

        expect(eventSource).toBe(lastEventSource);
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { renderHook } from 'test-utils/render';

import {
    getLastEventSource,
    MockEventSourceConstructor,
    resetMockEventSource,
    simulateSSEError,
    simulateSSEMessage,
    simulateSSEOpen,
} from '../test-utils/mock-event-source';
import { useSSE } from './use-sse.hook';

describe('useSSE', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('does not create an EventSource when url is undefined', () => {
        renderHook(() => useSSE(undefined, { onMessage: rs.fn() }));

        expect(MockEventSourceConstructor).not.toHaveBeenCalled();
    });

    it('creates an EventSource when url is provided', () => {
        renderHook(() => useSSE('/api/test/stream', { onMessage: rs.fn() }));

        expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1);
        expect(getLastEventSource().url).toContain('/api/test/stream');
    });

    it('closes the EventSource on unmount', () => {
        const { unmount } = renderHook(() => useSSE('/api/test/stream', { onMessage: rs.fn() }));
        const es = getLastEventSource();

        unmount();

        expect(es.close).toHaveBeenCalled();
    });

    it('calls onMessage when an SSE message arrives', () => {
        const onMessage = rs.fn();
        renderHook(() => useSSE<{ value: number }>('/api/test/stream', { onMessage }));
        const es = getLastEventSource();

        act(() => {
            simulateSSEMessage(es, { value: 42 });
        });

        expect(onMessage).toHaveBeenCalledWith({ value: 42 });
    });

    it('calls onOpen when the connection opens', () => {
        const onOpen = rs.fn();
        renderHook(() => useSSE('/api/test/stream', { onMessage: rs.fn(), onOpen }));
        const es = getLastEventSource();

        act(() => {
            simulateSSEOpen(es);
        });

        expect(onOpen).toHaveBeenCalledOnce();
    });

    it('calls onError and onClose when an error occurs', () => {
        const onError = rs.fn();
        const onClose = rs.fn();
        renderHook(() => useSSE('/api/test/stream', { onMessage: rs.fn(), onError, onClose }));
        const es = getLastEventSource();

        act(() => {
            simulateSSEError(es);
        });

        expect(onError).toHaveBeenCalledOnce();
        expect(onClose).toHaveBeenCalledOnce();
    });

    it('reconnects when url changes', () => {
        let url: string | undefined = '/api/test/1';
        const { rerender } = renderHook(() => useSSE(url, { onMessage: rs.fn() }));
        const firstEs = getLastEventSource();

        url = '/api/test/2';
        rerender();

        expect(firstEs.close).toHaveBeenCalled();
        expect(MockEventSourceConstructor).toHaveBeenCalledTimes(2);
        expect(getLastEventSource().url).toContain('/api/test/2');
    });

    it('closes when url becomes undefined', () => {
        let url: string | undefined = '/api/test/1';
        const { rerender } = renderHook(() => useSSE(url, { onMessage: rs.fn() }));
        const es = getLastEventSource();

        url = undefined;
        rerender();

        expect(es.close).toHaveBeenCalled();
    });

    it('returns a close function that closes the connection', () => {
        const onClose = rs.fn();
        const { result } = renderHook(() => useSSE('/api/test/stream', { onMessage: rs.fn(), onClose }));
        const es = getLastEventSource();

        act(() => {
            result.current.close();
        });

        expect(es.close).toHaveBeenCalled();
        expect(onClose).toHaveBeenCalledOnce();
    });

    it('uses latest callbacks without reconnecting', () => {
        let onMessage = rs.fn();
        const { rerender } = renderHook(() => useSSE('/api/test/stream', { onMessage }));
        const es = getLastEventSource();

        const secondOnMessage = rs.fn();
        onMessage = secondOnMessage;
        rerender();

        // Should NOT have created a new EventSource
        expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1);

        act(() => {
            simulateSSEMessage(es, { value: 1 });
        });

        expect(secondOnMessage).toHaveBeenCalledWith({ value: 1 });
    });
});

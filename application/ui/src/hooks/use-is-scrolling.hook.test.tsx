// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import { act, fireEvent, renderHook } from '@testing-library/react';

import { IsScrollingProvider, SCROLL_END_FALLBACK_MS, useIsScrolling } from './use-is-scrolling.hook';

describe('useIsScrolling', () => {
    const renderIsScrolling = () => {
        const scrollContainer = document.createElement('div');
        document.body.appendChild(scrollContainer);

        const wrapper = ({ children }: { children: ReactNode }) => (
            <IsScrollingProvider scrollRef={{ current: scrollContainer }}>{children}</IsScrollingProvider>
        );

        return { scrollContainer, ...renderHook(() => useIsScrolling(), { wrapper }) };
    };

    beforeEach(() => {
        vi.useFakeTimers();
    });

    afterEach(() => {
        vi.useRealTimers();
        document.body.innerHTML = '';
    });

    it('is false before any scrolling happens', () => {
        const { result } = renderIsScrolling();

        expect(result.current).toBe(false);
    });

    it('is true while the container is being scrolled', () => {
        const { scrollContainer, result } = renderIsScrolling();

        fireEvent.scroll(scrollContainer);

        expect(result.current).toBe(true);

        act(() => {
            vi.advanceTimersByTime(SCROLL_END_FALLBACK_MS - 1);
        });

        expect(result.current).toBe(true);
    });

    it('becomes false once the scroll end fallback timeout elapses', () => {
        const { scrollContainer, result } = renderIsScrolling();

        fireEvent.scroll(scrollContainer);

        act(() => {
            vi.advanceTimersByTime(SCROLL_END_FALLBACK_MS);
        });

        expect(result.current).toBe(false);
    });

    it('restarts the fallback timeout on every scroll event', () => {
        const { scrollContainer, result } = renderIsScrolling();

        fireEvent.scroll(scrollContainer);

        act(() => {
            vi.advanceTimersByTime(SCROLL_END_FALLBACK_MS - 1);
        });

        fireEvent.scroll(scrollContainer);

        act(() => {
            vi.advanceTimersByTime(SCROLL_END_FALLBACK_MS - 1);
        });

        expect(result.current).toBe(true);
    });

    it('becomes false as soon as scrollend is fired', () => {
        const { scrollContainer, result } = renderIsScrolling();

        fireEvent.scroll(scrollContainer);
        fireEvent(scrollContainer, new Event('scrollend'));

        expect(result.current).toBe(false);

        act(() => {
            vi.advanceTimersByTime(SCROLL_END_FALLBACK_MS);
        });

        expect(result.current).toBe(false);
    });
});

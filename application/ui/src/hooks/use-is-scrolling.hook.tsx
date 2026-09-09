// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, RefObject, useContext, useEffect, useRef, useState } from 'react';

import { useEventListener } from './event-listener.hook';

const IsScrollingContext = createContext(false);

// `scrollend` is not implemented by every webview (notably WKWebView), so a timeout is used as fallback.
export const SCROLL_END_FALLBACK_MS = 150;

type IsScrollingProviderProps = {
    scrollRef: RefObject<HTMLElement | null>;
    children: ReactNode;
};

export const IsScrollingProvider = ({ scrollRef, children }: IsScrollingProviderProps) => {
    const [isScrolling, setIsScrolling] = useState(false);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

    useEventListener(
        'scroll',
        () => {
            setIsScrolling(true);

            clearTimeout(timeoutRef.current);
            timeoutRef.current = setTimeout(() => setIsScrolling(false), SCROLL_END_FALLBACK_MS);
        },
        scrollRef
    );

    useEventListener(
        'scrollend',
        () => {
            clearTimeout(timeoutRef.current);
            setIsScrolling(false);
        },
        scrollRef
    );

    useEffect(() => () => clearTimeout(timeoutRef.current), []);

    return <IsScrollingContext.Provider value={isScrolling}>{children}</IsScrollingContext.Provider>;
};

export const useIsScrolling = (): boolean => useContext(IsScrollingContext);

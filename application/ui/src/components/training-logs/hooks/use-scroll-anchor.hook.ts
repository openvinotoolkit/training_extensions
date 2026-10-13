// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

export const useScrollAnchor = () => {
    const anchorRef = useRef<HTMLDivElement>(null);
    const [isAtBottom, setIsAtBottom] = useState(true);

    useEffect(() => {
        const anchor = anchorRef.current;

        if (!anchor) {
            return;
        }

        const observer = new IntersectionObserver(([entry]) => setIsAtBottom(entry.isIntersecting), {
            root: anchor.parentElement,
            threshold: 0,
        });

        observer.observe(anchor);

        return () => observer.disconnect();
    }, []);

    const scrollToBottom = () => {
        anchorRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    return { anchorRef, isAtBottom, scrollToBottom };
};

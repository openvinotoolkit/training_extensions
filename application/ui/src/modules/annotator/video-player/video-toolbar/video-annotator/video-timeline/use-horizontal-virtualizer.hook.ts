// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useRef, useState, type CSSProperties } from 'react';

import { useEventListener } from 'hooks/event-listener.hook';

export type VirtualItem = {
    index: number;
    size: number;
};

type HorizontalVirtualizerOptions = {
    itemCount: number;
    itemSize: number;
    viewportWidth: number;
    overscanCount: number;
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

/**
 * Windowing for a horizontally scrolling list of fixed-size items. Only the items in view (plus `overscanCount`
 * on each side) are returned; `innerStyle` shifts the element that renders them to their scroll position.
 */
export const useHorizontalVirtualizer = ({
    itemCount,
    itemSize,
    viewportWidth,
    overscanCount,
}: HorizontalVirtualizerOptions) => {
    const outerRef = useRef<HTMLDivElement>(null);
    const [scrollLeft, setScrollLeft] = useState(0);

    useEventListener('scroll', () => setScrollLeft(outerRef.current?.scrollLeft ?? 0), outerRef);

    const lastIndex = itemCount - 1;
    // Nothing is rendered until the viewport has been measured, otherwise we'd render an arbitrary window.
    const hasItems = itemCount > 0 && itemSize > 0 && viewportWidth > 0;

    const visibleStart = hasItems ? clamp(Math.floor(scrollLeft / itemSize), 0, lastIndex) : 0;
    const visibleStop = hasItems
        ? clamp(Math.ceil((scrollLeft + viewportWidth) / itemSize) - 1, visibleStart, lastIndex)
        : -1;
    const start = Math.max(visibleStart - overscanCount, 0);
    const stop = Math.min(visibleStop + overscanCount, lastIndex);

    const items: VirtualItem[] = hasItems
        ? Array.from({ length: stop - start + 1 }, (_, offset) => ({ index: start + offset, size: itemSize }))
        : [];

    const innerStyle: CSSProperties | undefined = hasItems
        ? { marginLeft: start * itemSize, width: (itemCount - start) * itemSize }
        : undefined;

    const scrollToItem = useCallback(
        (index: number) => {
            const maxScrollLeft = itemCount * itemSize - viewportWidth;

            if (outerRef.current === null || viewportWidth <= 0 || maxScrollLeft <= 0) {
                return;
            }

            const centeredScrollLeft = index * itemSize - (viewportWidth - itemSize) / 2;

            outerRef.current.scrollLeft = clamp(centeredScrollLeft, 0, maxScrollLeft);
        },
        [itemCount, itemSize, viewportWidth]
    );

    return { outerRef, items, innerStyle, scrollToItem };
};

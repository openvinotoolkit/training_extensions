// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from '@testing-library/react';

import { useHorizontalVirtualizer } from './use-horizontal-virtualizer.hook';

type HarnessProps = Parameters<typeof useHorizontalVirtualizer>[0] & { scrollToIndex?: number };

const Harness = ({ scrollToIndex, ...options }: HarnessProps) => {
    const { outerRef, items, innerStyle, scrollToItem } = useHorizontalVirtualizer(options);

    return (
        <div ref={outerRef} data-testid='outer'>
            <div style={innerStyle} data-testid='inner'>
                {items.map(({ index }) => (
                    <span key={index} data-testid='item'>
                        {index}
                    </span>
                ))}
            </div>
            <button onClick={() => scrollToIndex !== undefined && scrollToItem(scrollToIndex)}>scroll</button>
        </div>
    );
};

const renderHarness = (props: Partial<HarnessProps> = {}) => {
    const result = render(<Harness itemCount={100} itemSize={10} viewportWidth={50} overscanCount={2} {...props} />);
    const outer = screen.getByTestId('outer');

    // jsdom has no layout, so `scrollLeft` is made a plain writable value.
    Object.defineProperty(outer, 'scrollLeft', { value: 0, writable: true });

    return { ...result, outer };
};

const getRenderedIndices = () => screen.queryAllByTestId('item').map((item) => Number(item.textContent));

describe('useHorizontalVirtualizer', () => {
    it('renders only the visible items plus the overscan', () => {
        renderHarness();

        expect(getRenderedIndices()).toEqual([0, 1, 2, 3, 4, 5, 6]);
        expect(screen.getByTestId('inner')).toHaveStyle({ marginLeft: '0px', width: '1000px' });
    });

    it('shifts the rendered window when the container scrolls', () => {
        const { outer } = renderHarness();

        outer.scrollLeft = 200;
        fireEvent.scroll(outer);

        expect(getRenderedIndices()).toEqual([18, 19, 20, 21, 22, 23, 24, 25, 26]);
        expect(screen.getByTestId('inner')).toHaveStyle({ marginLeft: '180px', width: '820px' });
    });

    it.each([
        { index: 50, expected: 480 },
        { index: 0, expected: 0 },
        { index: 99, expected: 950 },
    ])('centers item $index in the viewport, clamped to the scrollable range', ({ index, expected }) => {
        const { outer } = renderHarness({ scrollToIndex: index });

        fireEvent.click(screen.getByRole('button', { name: 'scroll' }));

        expect(outer.scrollLeft).toBe(expected);
    });

    it('does not scroll when all items fit in the viewport', () => {
        const { outer } = renderHarness({ itemCount: 3, scrollToIndex: 2 });

        fireEvent.click(screen.getByRole('button', { name: 'scroll' }));

        expect(outer.scrollLeft).toBe(0);
        expect(getRenderedIndices()).toEqual([0, 1, 2]);
    });

    it.each([
        { itemCount: 0, itemSize: 10, viewportWidth: 50 },
        { itemCount: 10, itemSize: Number.NaN, viewportWidth: 50 },
        { itemCount: 10, itemSize: 10, viewportWidth: 0 },
    ])(
        'renders nothing for itemCount $itemCount, itemSize $itemSize and viewportWidth $viewportWidth',
        ({ itemCount, itemSize, viewportWidth }) => {
            renderHarness({ itemCount, itemSize, viewportWidth });

            expect(getRenderedIndices()).toEqual([]);
            expect(screen.getByTestId('inner')).not.toHaveAttribute('style');
        }
    );

    it('does not scroll before the viewport has been measured', () => {
        const { outer } = renderHarness({ viewportWidth: 0, scrollToIndex: 50 });

        fireEvent.click(screen.getByRole('button', { name: 'scroll' }));

        expect(outer.scrollLeft).toBe(0);
    });
});

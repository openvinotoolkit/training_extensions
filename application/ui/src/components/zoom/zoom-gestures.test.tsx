// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { useContainerSize } from './use-container-size';
import { ZoomTransform } from './zoom-transform';
import { ZoomProvider } from './zoom.provider';

vi.mock('./use-container-size', () => ({
    useContainerSize: vi.fn(),
}));

const INITIAL_SCALE = 0.9;
const MAX_SCALE = INITIAL_SCALE * 10;
const WHEEL_STEP = Math.exp(0.2);

// Stable references, otherwise useSyncZoom re-runs setZoom on every render and loops
const SCREEN_SIZE = { width: 500, height: 500 };
const CONTENT_SIZE = { width: 500, height: 500 };

const setupZoom = () => {
    vi.mocked(useContainerSize).mockImplementation(() => SCREEN_SIZE);

    render(
        <ZoomProvider>
            <ZoomTransform target={CONTENT_SIZE}>Content</ZoomTransform>
        </ZoomProvider>
    );

    const container = screen.getByTestId('zoom-transform').parentElement as HTMLElement;
    const getScale = () => Number(container.style.getPropertyValue('--zoom-scale'));

    return { container, getScale };
};

const wheel = (container: HTMLElement, deltaY: number, ctrlKey = false) =>
    fireEvent.wheel(container, { deltaY, ctrlKey, clientX: 250, clientY: 250 });

describe('Zoom gestures', () => {
    it('zooms in on wheel up and out on wheel down', () => {
        const { container, getScale } = setupZoom();

        wheel(container, -100);
        const zoomedIn = getScale();
        expect(zoomedIn).toBeGreaterThan(INITIAL_SCALE);

        wheel(container, 100);
        expect(getScale()).toBeLessThan(zoomedIn);
    });

    it('accumulates every wheel event when several arrive before a re-render', () => {
        const { container, getScale } = setupZoom();

        // One act, so React commits once for all five, the way a trackpad's stream outpaces rendering
        act(() => {
            for (let index = 0; index < 5; index++) {
                wheel(container, -100);
            }
        });

        // Reading a stale scale instead of the previous one would leave this at a single step
        expect(getScale()).toBeCloseTo(INITIAL_SCALE * WHEEL_STEP ** 5, 5);
    });

    it('keeps zooming in on a large delta instead of collapsing to fit-to-screen', () => {
        const { container, getScale } = setupZoom();

        // A factor of 1 - deltaY / 500 would turn negative here and clamp back to the initial scale
        wheel(container, -2000);

        expect(getScale()).toBeGreaterThan(INITIAL_SCALE);
    });

    it('never zooms outside the fit-to-screen and max bounds', () => {
        const { container, getScale } = setupZoom();

        act(() => {
            for (let index = 0; index < 20; index++) {
                wheel(container, -1000);
            }
        });
        expect(getScale()).toBeLessThanOrEqual(MAX_SCALE);

        act(() => {
            for (let index = 0; index < 20; index++) {
                wheel(container, 1000);
            }
        });
        expect(getScale()).toBeCloseTo(INITIAL_SCALE, 5);
    });

    it('does not apply the wheel factor to ctrl+wheel, which the pinch handler owns', () => {
        const { container, getScale } = setupZoom();

        wheel(container, -100, true);

        expect(getScale()).not.toBeCloseTo(INITIAL_SCALE * WHEEL_STEP, 5);
    });
});

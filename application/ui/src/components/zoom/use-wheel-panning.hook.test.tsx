// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent } from 'react';

import { act } from '@testing-library/react';
import { renderHook } from 'test-utils/render';

import { useWheelPanning } from './use-wheel-panning.hook';

const getMockedWheelPointerEvent = (data: Partial<PointerEvent<HTMLDivElement>>) => {
    return { clientX: 0, clientY: 0, ...data, button: 1 } as PointerEvent<HTMLDivElement>;
};

describe('useWheelPanning', () => {
    it('initialize with default values', () => {
        const { result } = renderHook(() => useWheelPanning(rs.fn()));
        expect(result.current.isGrabbing).toBe(false);
        expect(typeof result.current.onPointerDown).toBe('function');
        expect(typeof result.current.onPointerMove).toBe('function');
        expect(typeof result.current.onPointerUp).toBe('function');
    });

    it('set isGrabbing to true on pointer down', () => {
        const { result } = renderHook(() => useWheelPanning(rs.fn()));
        act(() => {
            result.current.onPointerDown(getMockedWheelPointerEvent({ clientX: 10, clientY: 20 }));
        });

        expect(result.current.isGrabbing).toBe(true);
    });

    it('update position on pointer move while panning', async () => {
        const mockedSetPosition = rs.fn();

        const { result } = renderHook(() => useWheelPanning(rs.fn()));

        act(() => {
            result.current.onPointerDown(getMockedWheelPointerEvent({ clientX: 10, clientY: 10 }));

            result.current.onPointerMove(mockedSetPosition)(getMockedWheelPointerEvent({ clientX: 10, clientY: 10 }));
        });

        expect(result.current.isGrabbing).toBe(true);
        expect(mockedSetPosition).toHaveBeenCalledWith({ x: 0, y: 0 });
    });

    it('set isPanning to false on pointer up', () => {
        const mockedSetPanning = rs.fn();
        const { result } = renderHook(() => useWheelPanning(mockedSetPanning));
        act(() => {
            result.current.onPointerDown(getMockedWheelPointerEvent({ clientX: 10, clientY: 20 }));
        });
        expect(result.current.isGrabbing).toBe(true);
        expect(mockedSetPanning).toHaveBeenCalledWith(true);

        act(() => {
            result.current.onPointerUp();
        });
        expect(result.current.isGrabbing).toBe(false);
        expect(mockedSetPanning).toHaveBeenCalledWith(false);
    });
});

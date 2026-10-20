// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { getMockedVideoFrame } from 'mocks/mock-media';
import { renderHook } from 'test-utils/render';

import { useVideoControls } from './use-video-controls';

const makeVideoRef = (overrides: Partial<HTMLVideoElement> = {}) => {
    return {
        current: {
            currentTime: 0,
            play: rs.fn().mockResolvedValue(undefined),
            pause: rs.fn(),
            ...overrides,
        } as unknown as HTMLVideoElement,
    };
};

const FPS = 60;
const FRAME_COUNT = 600;
const STEP = 1;

const renderControls = ({
    frameNumber = 0,
    step = STEP,
    videoRef = makeVideoRef(),
    selectVideoFrame = rs.fn(),
    changeCurrentFrameIndex = rs.fn(),
} = {}) => {
    const videoFrame = getMockedVideoFrame({
        fps: FPS,
        frame_count: FRAME_COUNT,
        frame_number: frameNumber,
        frame_stride: STEP,
    });

    return renderHook(() =>
        useVideoControls({
            step,
            videoRef,
            videoFrame,
            selectVideoFrame,
            changeCurrentFrameIndex,
        })
    );
};

describe('useVideoControls', () => {
    describe('play', () => {
        it('calls videoRef.play() and sets isPlaying to true', async () => {
            const videoRef = makeVideoRef();
            const { result } = renderControls({ videoRef });

            await act(async () => {
                await result.current.play();
            });

            expect(videoRef.current.play).toHaveBeenCalledTimes(1);
            expect(result.current.isPlaying).toBe(true);
        });

        it('sets isPlaying back to false if videoRef.play() rejects', async () => {
            const videoRef = makeVideoRef({ play: rs.fn().mockRejectedValue(new Error('AbortError')) });
            const { result } = renderControls({ videoRef });

            await act(async () => {
                await result.current.play();
            });

            expect(result.current.isPlaying).toBe(false);
        });

        it('does nothing when videoRef.current is null', async () => {
            const videoRef = { current: null };
            // @ts-expect-error - testing current null edge case
            const { result } = renderControls({ videoRef });

            await act(async () => {
                await result.current.play();
            });

            expect(result.current.isPlaying).toBe(false);
        });
    });

    describe('pause', () => {
        it('calls videoRef.pause() and sets isPlaying to false', async () => {
            const videoRef = makeVideoRef();
            const { result } = renderControls({ videoRef });

            await act(async () => {
                await result.current.play();
            });

            act(() => {
                result.current.pause();
            });

            expect(videoRef.current.pause).toHaveBeenCalled();
            expect(result.current.isPlaying).toBe(false);
        });

        it('snaps to the nearest step-aligned frame on pause', async () => {
            const selectVideoFrame = rs.fn();
            const changeCurrentFrameIndex = rs.fn();
            const videoRef = makeVideoRef();

            const { result } = renderControls({
                frameNumber: 70,
                step: 60,
                videoRef,
                selectVideoFrame,
                changeCurrentFrameIndex,
            });

            await act(async () => {
                await result.current.play();
            });

            act(() => {
                result.current.pause();
            });

            expect(selectVideoFrame).toHaveBeenCalledWith(expect.objectContaining({ frame_number: 60 }));
        });

        it('does nothing when videoRef.current is null', () => {
            const videoRef = { current: null };
            const selectVideoFrame = rs.fn();
            // @ts-expect-error - testing current null edge case
            const { result } = renderControls({ videoRef, selectVideoFrame });

            act(() => {
                result.current.pause();
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });
    });

    describe('nextFrame', () => {
        it('calls selectVideoFrame with the next frame number when not playing', () => {
            const selectVideoFrame = rs.fn();
            const changeCurrentFrameIndex = rs.fn();
            const videoRef = makeVideoRef();
            const step = 60;
            const frameNumber = 0;
            const { result } = renderControls({
                frameNumber,
                step,
                videoRef,
                selectVideoFrame,
                changeCurrentFrameIndex,
            });

            act(() => {
                result.current.nextFrame();
            });

            expect(selectVideoFrame).toHaveBeenCalledWith(
                expect.objectContaining({ frame_number: frameNumber + step })
            );
            expect(changeCurrentFrameIndex).toHaveBeenCalledWith(frameNumber + step);
        });

        it('advances videoRef.currentTime when playing instead of selecting a frame', async () => {
            const selectVideoFrame = rs.fn();
            const changeCurrentFrameIndex = rs.fn();
            const videoRef = makeVideoRef();
            const { result } = renderControls({
                frameNumber: 0,
                step: 1,
                videoRef,
                selectVideoFrame,
                changeCurrentFrameIndex,
            });

            await act(async () => {
                await result.current.play();
            });

            // reset call counts after play
            selectVideoFrame.mockClear();
            changeCurrentFrameIndex.mockClear();

            act(() => {
                result.current.nextFrame();
            });

            expect(videoRef.current.currentTime).toBe(1);
            expect(changeCurrentFrameIndex).toHaveBeenCalledWith(1);
            expect(selectVideoFrame).not.toHaveBeenCalled();
        });

        it('does nothing when already at the last frame', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = makeVideoRef();
            // last valid frame for step=1 is FRAME_COUNT - 1 = 599
            const { result } = renderControls({ frameNumber: FRAME_COUNT - 1, videoRef, selectVideoFrame });

            act(() => {
                result.current.nextFrame();
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });

        it('does nothing when videoRef.current is null', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = { current: null };
            // @ts-expect-error - testing current null edge case
            const { result } = renderControls({ videoRef, selectVideoFrame });

            act(() => {
                result.current.nextFrame();
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });
    });

    describe('previousFrame', () => {
        it('calls selectVideoFrame with the previous frame number when not playing', () => {
            const selectVideoFrame = rs.fn();
            const changeCurrentFrameIndex = rs.fn();
            const videoRef = makeVideoRef();
            const step = 1;
            const frameNumber = 5;

            const { result } = renderControls({
                frameNumber,
                step,
                videoRef,
                selectVideoFrame,
                changeCurrentFrameIndex,
            });

            act(() => {
                result.current.previousFrame();
            });

            expect(selectVideoFrame).toHaveBeenCalledWith(
                expect.objectContaining({ frame_number: frameNumber - step })
            );
            expect(changeCurrentFrameIndex).toHaveBeenCalledWith(frameNumber - step);
        });

        it('rewinds videoRef.currentTime when playing instead of selecting a frame', async () => {
            const selectVideoFrame = rs.fn();
            const changeCurrentFrameIndex = rs.fn();
            const ONE_SECOND = 1;
            const currentTime = 10;
            const step = 1;
            const frameNumber = 5;
            const videoRef = makeVideoRef({ currentTime });
            const { result } = renderControls({
                frameNumber,
                step,
                videoRef,
                selectVideoFrame,
                changeCurrentFrameIndex,
            });

            await act(async () => {
                await result.current.play();
            });

            selectVideoFrame.mockClear();
            changeCurrentFrameIndex.mockClear();

            act(() => {
                result.current.previousFrame();
            });

            expect(videoRef.current.currentTime).toBe(currentTime - ONE_SECOND);
            expect(changeCurrentFrameIndex).toHaveBeenCalledWith(frameNumber - step);
            expect(selectVideoFrame).not.toHaveBeenCalled();
        });

        it('does nothing when already at the first frame', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = makeVideoRef();
            const { result } = renderControls({ frameNumber: 0, videoRef, selectVideoFrame });

            act(() => {
                result.current.previousFrame();
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });

        it('does nothing when videoRef.current is null', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = { current: null };
            // @ts-expect-error - testing current null edge case
            const { result } = renderControls({ frameNumber: 5, videoRef, selectVideoFrame });

            act(() => {
                result.current.previousFrame();
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });
    });

    describe('goto', () => {
        it('seeks to the given frame when not playing', () => {
            const selectVideoFrame = rs.fn();
            const changeCurrentFrameIndex = rs.fn();
            const videoRef = makeVideoRef();
            const { result } = renderControls({ videoRef, selectVideoFrame, changeCurrentFrameIndex });

            act(() => {
                result.current.goto(120);
            });

            expect(selectVideoFrame).toHaveBeenCalledWith(expect.objectContaining({ frame_number: 120 }));
            expect(changeCurrentFrameIndex).toHaveBeenCalledWith(120);
            expect(videoRef.current.currentTime).toBe(121 / FPS);
        });

        it('pauses and seeks when called while playing', async () => {
            const selectVideoFrame = rs.fn();
            const videoRef = makeVideoRef();
            const { result } = renderControls({ videoRef, selectVideoFrame });

            await act(async () => {
                await result.current.play();
            });

            act(() => {
                result.current.goto(60);
            });

            expect(videoRef.current.pause).toHaveBeenCalled();
            expect(result.current.isPlaying).toBe(false);
            expect(selectVideoFrame).toHaveBeenCalledWith(expect.objectContaining({ frame_number: 60 }));
        });

        it('snaps to the nearest step-aligned frame', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = makeVideoRef();
            const { result } = renderControls({ frameNumber: 0, step: 60, videoRef, selectVideoFrame });

            act(() => {
                result.current.goto(70);
            });

            expect(selectVideoFrame).toHaveBeenCalledWith(expect.objectContaining({ frame_number: 60 }));
        });

        it('does nothing when the frame number is out of range (>= totalFrames)', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = makeVideoRef();
            const { result } = renderControls({ videoRef, selectVideoFrame });

            act(() => {
                result.current.goto(FRAME_COUNT);
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });

        it('does nothing when the frame number is negative', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = makeVideoRef();
            const { result } = renderControls({ videoRef, selectVideoFrame });

            act(() => {
                result.current.goto(-1);
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });

        it('does nothing when videoRef.current is null', () => {
            const selectVideoFrame = rs.fn();
            const videoRef = { current: null };
            // @ts-expect-error - testing current null edge case
            const { result } = renderControls({ videoRef, selectVideoFrame });

            act(() => {
                result.current.goto(60);
            });

            expect(selectVideoFrame).not.toHaveBeenCalled();
        });
    });

    describe('canSelectNextFrame / canSelectPreviousFrame', () => {
        it('canSelectNextFrame is true when not at the last frame', () => {
            const { result } = renderControls({ frameNumber: 0 });
            expect(result.current.canSelectNextFrame).toBe(true);
        });

        it('canSelectNextFrame is false when at the last frame', () => {
            const { result } = renderControls({ frameNumber: FRAME_COUNT - 1 });
            expect(result.current.canSelectNextFrame).toBe(false);
        });

        it('canSelectPreviousFrame is true when not at the first frame', () => {
            const { result } = renderControls({ frameNumber: 1 });
            expect(result.current.canSelectPreviousFrame).toBe(true);
        });

        it('canSelectPreviousFrame is false when at the first frame', () => {
            const { result } = renderControls({ frameNumber: 0 });
            expect(result.current.canSelectPreviousFrame).toBe(false);
        });
    });
});

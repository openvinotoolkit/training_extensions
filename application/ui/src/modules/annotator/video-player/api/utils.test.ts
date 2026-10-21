// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getVideoFrameRangeIndexes } from './utils';

describe('getVideoFrameRangeIndexes', () => {
    describe('startFrameIndex', () => {
        it('returns 0 as startFrameIndex when frameNumber is 0', () => {
            const result = getVideoFrameRangeIndexes({ frameNumber: 0, frames: 100, frameSkip: 1 });

            expect(result.startFrameIndex).toBe(0);
        });

        it('returns 0 as startFrameIndex when frameNumber is within the first chunk', () => {
            const result = getVideoFrameRangeIndexes({ frameNumber: 15, frames: 100, frameSkip: 1 });

            expect(result.startFrameIndex).toBe(0);
        });

        it('returns the chunk boundary as startFrameIndex when frameNumber is exactly at a chunk boundary', () => {
            // chunkSize=30, frameSkip=1 → annotationChunkSize=30
            const result = getVideoFrameRangeIndexes({ frameNumber: 30, frames: 100, frameSkip: 1 });

            expect(result.startFrameIndex).toBe(30);
        });

        it('returns the correct startFrameIndex when frameNumber is in the middle of a later chunk', () => {
            // chunkSize=30, frameSkip=1 → annotationChunkSize=30; frame 45 falls in chunk [30, 59]
            const result = getVideoFrameRangeIndexes({ frameNumber: 45, frames: 100, frameSkip: 1 });

            expect(result.startFrameIndex).toBe(30);
        });

        it('returns a startFrameIndex scaled by frameSkip', () => {
            // chunkSize=30, frameSkip=60 → annotationChunkSize=1800; frame 1900 falls in chunk [1800, 3599]
            const result = getVideoFrameRangeIndexes({ frameNumber: 1900, frames: 5000, frameSkip: 60 });

            expect(result.startFrameIndex).toBe(1800);
        });

        it('returns 0 as startFrameIndex for a custom chunkSize when frameNumber is within the first chunk', () => {
            // chunkSize=10, frameSkip=1 → annotationChunkSize=10; frame 5 is in [0, 9]
            const result = getVideoFrameRangeIndexes({ frameNumber: 5, frames: 100, frameSkip: 1, chunkSize: 10 });

            expect(result.startFrameIndex).toBe(0);
        });

        it('returns the correct startFrameIndex for a custom chunkSize', () => {
            // chunkSize=10, frameSkip=1 → annotationChunkSize=10; frame 25 falls in chunk [20, 29]
            const result = getVideoFrameRangeIndexes({ frameNumber: 25, frames: 100, frameSkip: 1, chunkSize: 10 });

            expect(result.startFrameIndex).toBe(20);
        });
    });

    describe('endFrameIndex', () => {
        it('returns annotationChunkSize - 1 as endFrameIndex when the chunk does not exceed total frames', () => {
            // chunkSize=30, frameSkip=1 → annotationChunkSize=30; chunk [0, 29], frames=100
            const result = getVideoFrameRangeIndexes({ frameNumber: 0, frames: 100, frameSkip: 1 });

            expect(result.endFrameIndex).toBe(29);
        });

        it('returns frames as endFrameIndex when the chunk would exceed total frames', () => {
            // chunkSize=30, frameSkip=1 → annotationChunkSize=30; chunk [90, 119] clipped to [90, 100]
            const result = getVideoFrameRangeIndexes({ frameNumber: 95, frames: 100, frameSkip: 1 });

            expect(result.endFrameIndex).toBe(100);
        });

        it('returns frames as endFrameIndex when frameNumber equals frames', () => {
            const result = getVideoFrameRangeIndexes({ frameNumber: 100, frames: 100, frameSkip: 1 });

            expect(result.endFrameIndex).toBe(100);
        });

        it('returns the correct endFrameIndex scaled by frameSkip', () => {
            // chunkSize=30, frameSkip=60 → annotationChunkSize=1800; chunk [0, 1799], frames=5000
            const result = getVideoFrameRangeIndexes({ frameNumber: 0, frames: 5000, frameSkip: 60 });

            expect(result.endFrameIndex).toBe(1799);
        });

        it('returns the correct endFrameIndex for a custom chunkSize', () => {
            // chunkSize=5, frameSkip=3 → annotationChunkSize=15; chunk [0, 14], frames=100
            const result = getVideoFrameRangeIndexes({ frameNumber: 0, frames: 100, frameSkip: 3, chunkSize: 5 });

            expect(result.endFrameIndex).toBe(14);
        });
    });

    describe('combined startFrameIndex and endFrameIndex', () => {
        it('returns the correct range for a mid-video frame with default chunkSize', () => {
            // chunkSize=30, frameSkip=1 → annotationChunkSize=30; frame 60 is in [60, 89]
            const result = getVideoFrameRangeIndexes({ frameNumber: 60, frames: 200, frameSkip: 1 });

            expect(result).toEqual({ startFrameIndex: 60, endFrameIndex: 89 });
        });

        it('returns a single-frame range when chunkSize is 1 and frameSkip is 1', () => {
            // annotationChunkSize=1; frame 7 is in [7, 7]
            const result = getVideoFrameRangeIndexes({ frameNumber: 7, frames: 100, frameSkip: 1, chunkSize: 1 });

            expect(result).toEqual({ startFrameIndex: 7, endFrameIndex: 7 });
        });

        it('returns the correct range when chunkSize is 1 and the chunk is at the end of the video', () => {
            // annotationChunkSize=1; frame 100 clipped to frames=100
            const result = getVideoFrameRangeIndexes({ frameNumber: 100, frames: 100, frameSkip: 1, chunkSize: 1 });

            expect(result).toEqual({ startFrameIndex: 100, endFrameIndex: 100 });
        });

        it('returns the correct range for a large frameSkip value', () => {
            // chunkSize=30, frameSkip=60 → annotationChunkSize=1800; frame 0 is in [0, 1799] clipped to [0, 500]
            const result = getVideoFrameRangeIndexes({ frameNumber: 0, frames: 2000, frameSkip: 60 });

            expect(result).toEqual({ startFrameIndex: 0, endFrameIndex: 1799 });
        });
    });
});

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Point, Shape } from '../../../../shared/types';

export interface InteractiveAnnotationPoint extends Point {
    positive: boolean;
}

export type SegmentAnythingDecodingResult = {
    shapes: Shape[];
    // Raw low resolution prediction, fed back to the decoder as `previousMask` to refine the same object.
    lowResMask?: Float32Array;
};

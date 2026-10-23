// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { SegmentAnythingOptions } from './segment-anything-options.component';

describe('SegmentAnythingOptions', () => {
    it('explains how to include and exclude areas', () => {
        render(<SegmentAnythingOptions />);

        expect(
            screen.getByText(
                'Shift + click to include an area, Alt + click to exclude it. Click to create the annotation, Esc to clear the points.'
            )
        ).toBeVisible();
    });
});

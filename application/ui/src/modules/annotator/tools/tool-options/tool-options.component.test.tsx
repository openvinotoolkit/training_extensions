// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import type { ToolType } from '../../tool-type';
import { ToolOptions } from './tool-options.component';

let mockActiveTool: ToolType | null = 'ssim';

vi.mock('../../tool-provider.component', () => ({
    useTool: () => ({ activeTool: mockActiveTool, setActiveTool: vi.fn() }),
}));

vi.mock('../ssim-tool/ssim-options.component', () => ({
    SSIMOptions: () => <span>ssim options</span>,
}));

describe('ToolOptions', () => {
    it('renders the SSIM options while the SSIM tool is active', () => {
        mockActiveTool = 'ssim';
        render(<ToolOptions />);

        expect(screen.getByRole('group', { name: 'tool options' })).toHaveTextContent('ssim options');
    });

    it.each<ToolType | null>(['bounding-box', 'polygon', 'sam', null])('renders nothing for %s', (tool) => {
        mockActiveTool = tool;
        render(<ToolOptions />);

        expect(screen.queryByRole('group', { name: 'tool options' })).not.toBeInTheDocument();
    });
});

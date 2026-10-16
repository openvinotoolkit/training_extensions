// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { SSIMOptions } from './ssim-options.component';

const mockAccept = vi.fn();
const mockReject = vi.fn();
let mockState = { hasPreview: false, isProcessing: false };

vi.mock('./ssim-state-provider.component', () => ({
    useSSIMState: () => ({ ...mockState, accept: mockAccept, reject: mockReject }),
}));

describe('SSIMOptions', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockState = { hasPreview: false, isProcessing: false };
    });

    it('disables accept and reject until a template is drawn', async () => {
        render(<SSIMOptions />);

        expect(await screen.findByRole('button', { name: 'Accept' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Reject' })).toBeDisabled();

        await userEvent.keyboard('{Enter}');
        await userEvent.keyboard('{Escape}');

        expect(mockAccept).not.toHaveBeenCalled();
        expect(mockReject).not.toHaveBeenCalled();
    });

    it('accepts and rejects the preview with the buttons', async () => {
        mockState = { hasPreview: true, isProcessing: false };
        render(<SSIMOptions />);

        fireEvent.click(await screen.findByRole('button', { name: 'Accept' }));
        fireEvent.click(screen.getByRole('button', { name: 'Reject' }));

        expect(mockAccept).toHaveBeenCalledTimes(1);
        expect(mockReject).toHaveBeenCalledTimes(1);
    });

    it('handles Enter and Escape before other listeners can submit or close the annotator', async () => {
        mockState = { hasPreview: true, isProcessing: false };
        const documentKeyDown = vi.fn();
        document.addEventListener('keydown', documentKeyDown);

        render(<SSIMOptions />);
        await screen.findByRole('button', { name: 'Accept' });

        await userEvent.keyboard('{Enter}');
        await userEvent.keyboard('{Escape}');

        expect(mockAccept).toHaveBeenCalledTimes(1);
        expect(mockReject).toHaveBeenCalledTimes(1);
        expect(documentKeyDown).not.toHaveBeenCalled();

        document.removeEventListener('keydown', documentKeyDown);
    });

    it('only allows rejecting while a run is in flight', async () => {
        mockState = { hasPreview: false, isProcessing: true };
        render(<SSIMOptions />);

        expect(await screen.findByRole('button', { name: 'Accept' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Reject' })).toBeEnabled();

        await userEvent.keyboard('{Enter}');
        await userEvent.keyboard('{Escape}');

        expect(mockAccept).not.toHaveBeenCalled();
        expect(mockReject).toHaveBeenCalledTimes(1);
    });
});

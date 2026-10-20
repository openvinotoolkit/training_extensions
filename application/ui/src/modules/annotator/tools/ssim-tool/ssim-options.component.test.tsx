// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { SSIMOptions } from './ssim-options.component';

const mockAccept = rs.fn();
const mockReject = rs.fn();
const mockSetNumberOfItems = rs.fn();
const mockSetAutoMergeDuplicates = rs.fn();
let mockState = { hasPreview: false, isProcessing: false, toolState: { matches: [] as unknown[], threshold: 0 } };

const withPreview = (numberOfMatches: number, threshold: number) => ({
    hasPreview: true,
    isProcessing: false,
    toolState: { matches: Array.from({ length: numberOfMatches }), threshold },
});

rs.mock('./ssim-state-provider.component', () => ({
    useSSIMState: () => ({
        ...mockState,
        accept: mockAccept,
        reject: mockReject,
        setNumberOfItems: mockSetNumberOfItems,
        autoMergeDuplicates: true,
        setAutoMergeDuplicates: mockSetAutoMergeDuplicates,
    }),
}));

describe('SSIMOptions', () => {
    beforeEach(() => {
        rs.clearAllMocks();
        mockState = { hasPreview: false, isProcessing: false, toolState: { matches: [], threshold: 0 } };
    });

    it('disables accept and reject until a template is drawn', async () => {
        render(<SSIMOptions />);

        expect(await screen.findByRole('button', { name: 'Accept' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Reject' })).toBeDisabled();
        expect(screen.getByRole('slider', { name: 'Number of items' })).toBeDisabled();
        expect(screen.getByRole('switch', { name: 'Auto-merge duplicates' })).toBeEnabled();

        await userEvent.keyboard('{Enter}');
        await userEvent.keyboard('{Escape}');

        expect(mockAccept).not.toHaveBeenCalled();
        expect(mockReject).not.toHaveBeenCalled();
    });

    it('accepts and rejects the preview with the buttons', async () => {
        mockState = withPreview(3, 2);
        render(<SSIMOptions />);

        fireEvent.click(await screen.findByRole('button', { name: 'Accept' }));
        fireEvent.click(screen.getByRole('button', { name: 'Reject' }));

        expect(mockAccept).toHaveBeenCalledTimes(1);
        expect(mockReject).toHaveBeenCalledTimes(1);
    });

    it('handles Enter and Escape before other listeners can submit or close the annotator', async () => {
        mockState = withPreview(3, 2);
        const documentKeyDown = rs.fn();
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

    it('only allows rejecting while the first run is in flight', async () => {
        mockState = { ...withPreview(3, 2), hasPreview: false, isProcessing: true };
        render(<SSIMOptions />);

        expect(await screen.findByRole('button', { name: 'Accept' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Reject' })).toBeEnabled();
        expect(screen.getByRole('slider', { name: 'Number of items' })).toBeDisabled();
        expect(screen.getByRole('switch', { name: 'Auto-merge duplicates' })).toBeEnabled();

        await userEvent.keyboard('{Enter}');
        await userEvent.keyboard('{Escape}');

        expect(mockAccept).not.toHaveBeenCalled();
        expect(mockReject).toHaveBeenCalledTimes(1);
    });

    it('keeps the controls enabled while re-running with a preview on screen', async () => {
        mockState = { ...withPreview(3, 2), isProcessing: true };
        render(<SSIMOptions />);

        expect(await screen.findByRole('button', { name: 'Accept' })).toBeEnabled();
        expect(screen.getByRole('button', { name: 'Reject' })).toBeEnabled();
        expect(screen.getByRole('slider', { name: 'Number of items' })).toBeEnabled();
        expect(screen.getByRole('switch', { name: 'Auto-merge duplicates' })).toBeEnabled();
    });

    it('changes the number of items with the slider', async () => {
        mockState = withPreview(5, 3);
        render(<SSIMOptions />);

        const slider = await screen.findByRole('slider', { name: 'Number of items' });

        expect(slider).toHaveValue('3');

        fireEvent.keyDown(slider, { key: 'ArrowRight' });

        expect(mockSetNumberOfItems).toHaveBeenCalledWith(4);
    });

    it('disables the slider when only the template was found', async () => {
        mockState = withPreview(1, 1);
        render(<SSIMOptions />);

        expect(await screen.findByRole('slider', { name: 'Number of items' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Accept' })).toBeEnabled();
    });

    it('toggles auto-merge duplicates', async () => {
        render(<SSIMOptions />);

        fireEvent.click(await screen.findByRole('switch', { name: 'Auto-merge duplicates' }));

        expect(mockSetAutoMergeDuplicates).toHaveBeenCalledWith(false);
    });

    it('rejects with Escape while the slider is focused', async () => {
        mockState = withPreview(5, 3);
        render(<SSIMOptions />);

        (await screen.findByRole('slider', { name: 'Number of items' })).focus();
        await userEvent.keyboard('{Escape}');

        expect(mockReject).toHaveBeenCalledTimes(1);
    });

    it('ignores Enter and Escape while typing in a text field', async () => {
        mockState = withPreview(5, 3);
        render(
            <>
                <input aria-label='search' />
                <SSIMOptions />
            </>
        );

        (await screen.findByRole('textbox', { name: 'search' })).focus();
        await userEvent.keyboard('{Enter}');
        await userEvent.keyboard('{Escape}');

        expect(mockAccept).not.toHaveBeenCalled();
        expect(mockReject).not.toHaveBeenCalled();
    });
});

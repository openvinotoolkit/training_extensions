// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { getUploadPercentage, UploadProgress } from './upload-progress.component';

describe('UploadProgress', () => {
    it('shows the transferred bytes and the percentage', () => {
        render(<UploadProgress bytesSent={1_500_000} bytesTotal={6_000_000} />);

        expect(screen.getByRole('progressbar', { name: 'Upload progress' })).toHaveAttribute('aria-valuenow', '25');
        expect(screen.getByText('1.5 MB of 6 MB')).toBeVisible();
        expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
    });

    it('calls onCancel when the cancel button is pressed', async () => {
        const onCancel = vi.fn();
        render(<UploadProgress bytesSent={1} bytesTotal={2} onCancel={onCancel} />);

        await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));

        expect(onCancel).toHaveBeenCalledOnce();
    });

    it.each([
        [0, 0, 0],
        [5, 10, 50],
        [20, 10, 100],
    ])('computes %i of %i bytes as %i%%', (bytesSent, bytesTotal, expected) => {
        expect(getUploadPercentage(bytesSent, bytesTotal)).toBe(expected);
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { FilterChips } from './filter-chips.component';

describe('FilterChips', () => {
    it('renders the given name', () => {
        render(<FilterChips name={'Cat'} ariaLabel={'Remove Cat filter'} onClose={vi.fn()} />);

        expect(screen.getByText('Cat')).toBeVisible();
    });

    it('calls onClose when the close icon is clicked', () => {
        const mockOnClose = vi.fn();
        render(<FilterChips name={'Cat'} ariaLabel={'Remove Cat filter'} onClose={mockOnClose} />);

        fireEvent.click(screen.getByRole('button', { name: 'Remove Cat filter' }));

        expect(mockOnClose).toHaveBeenCalledTimes(1);
    });
});

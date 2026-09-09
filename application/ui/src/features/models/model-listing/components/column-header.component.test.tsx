// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import type { SortDescriptor } from '../types';
import { ColumnHeader } from './column-header.component';

describe('ColumnHeader', () => {
    const renderColumnHeader = (sortBy: SortDescriptor, onSortChange = vi.fn()) => {
        render(<ColumnHeader label={'Total size'} sortKey={'size'} sortBy={sortBy} onSortChange={onSortChange} />);

        return onSortChange;
    };

    it('sorts by the column when clicked', async () => {
        const onSortChange = renderColumnHeader({ key: 'name', direction: 'asc' });

        fireEvent.click(screen.getByRole('button', { name: 'Sort by Total size' }));

        expect(onSortChange).toHaveBeenCalledWith('size');
    });

    it('exposes the direction when the column is sorted', () => {
        renderColumnHeader({ key: 'size', direction: 'desc' });

        expect(screen.getByRole('button', { name: 'Total size, sorted descending' })).toBeInTheDocument();
    });
});

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMultipleMockedMediaImage } from 'mocks/mock-media';
import { Size } from 'react-aria-components';
import { render } from 'test-utils/render';

import { VirtualizerGridLayout } from './virtualizer-grid-layout.component';

// required configuration; otherwise, the list renders empty
const mockedLayoutOptions = {
    maxColumns: 1,
    maxItemSize: new Size(100, 100),
    maxHorizontalSpace: 1000,
};

describe('VirtualizerGridLayout', () => {
    const mockedItems = getMultipleMockedMediaImage(2, 'test');

    it('renders all items as visible options', () => {
        const mockedLoadingMore = rs.fn();

        render(
            <VirtualizerGridLayout
                items={mockedItems}
                ariaLabel={'test list'}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={false}
                onLoadMore={mockedLoadingMore}
                contentItem={(item) => <div>{item.name}</div>}
            />
        );

        expect(mockedLoadingMore).toHaveBeenCalledOnce();
        mockedItems.forEach((item) => expect(screen.getByRole('option', { name: item.name })).toBeVisible());
    });

    it('renders with empty items', () => {
        render(
            <VirtualizerGridLayout
                items={[]}
                ariaLabel={'empty list'}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={false}
                onLoadMore={rs.fn()}
                contentItem={() => <div></div>}
            />
        );
        expect(screen.queryAllByRole('option')).toHaveLength(0);
    });

    it('renders loading indicator when isLoadingMore is true', () => {
        const mockedLoadingMore = rs.fn();
        render(
            <VirtualizerGridLayout
                items={mockedItems}
                ariaLabel={'loading list'}
                selectionMode={'single'}
                layoutOptions={mockedLayoutOptions}
                isLoadingMore={true}
                onLoadMore={mockedLoadingMore}
                contentItem={(item) => <div>{item.name}</div>}
            />
        );

        expect(mockedLoadingMore).not.toHaveBeenCalled();
        expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
});

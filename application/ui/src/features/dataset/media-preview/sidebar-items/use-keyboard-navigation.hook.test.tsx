// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { fireEvent, screen } from '@testing-library/react';
import { getMultipleMockedMediaImage } from 'mocks/mock-media';
import { render } from 'test-utils/render';

import { useKeyboardNavigation, type UseKeyboardNavigationProps } from './use-keyboard-navigation.hook';

const App = ({ ...options }: Omit<UseKeyboardNavigationProps, 'ref'>) => {
    const ref = useRef(null);
    useKeyboardNavigation({ ...options, ref });

    return <div ref={ref} data-testid='target' />;
};

describe('useKeyboardNavigation', () => {
    const items = getMultipleMockedMediaImage(3);

    it('calls onSelectedMediaItem with previous item on ArrowUp', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={1} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowUp', code: 'ArrowUp' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[0]);
    });

    it('calls onSelectedMediaItem with previous item on ArrowLeft', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={1} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowLeft', code: 'ArrowLeft' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[0]);
    });

    it('calls onSelectedMediaItem with next item on ArrowDown', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={1} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowDown', code: 'ArrowDown' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[2]);
    });

    it('calls onSelectedMediaItem with next item on ArrowRight', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={1} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowRight', code: 'ArrowRight' });
        expect(mockedOnSelectedMediaItem).toHaveBeenCalledWith(items[2]);
    });

    it('does not go below index 0 on ArrowUp or ArrowLeft', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={0} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowUp', code: 'ArrowUp' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowLeft', code: 'ArrowLeft' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });

    it('does not go above last index on ArrowDown or ArrowRight', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={2} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowDown', code: 'ArrowDown' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowRight', code: 'ArrowRight' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });

    it('does not call onSelectedMediaItem for other keys', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={items} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={1} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'Enter' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });

    it('does nothing if items is empty', async () => {
        const mockedOnSelectedMediaItem = rs.fn();
        render(<App items={[]} onSelectedMediaItem={mockedOnSelectedMediaItem} selectedIndex={0} />);

        fireEvent.keyDown(screen.getByTestId('target'), { key: 'ArrowDown' });
        expect(mockedOnSelectedMediaItem).not.toHaveBeenCalled();
    });
});

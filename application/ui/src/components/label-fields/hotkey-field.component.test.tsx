// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { HotkeyField } from './hotkey-field.component';

describe('HotkeyField', () => {
    const mockOnHotkeyChange = rs.fn();

    const renderHotkeyField = (hotkey: string | null | undefined = null, errorMessage?: string) => {
        render(<HotkeyField hotkey={hotkey} onHotkeyChange={mockOnHotkeyChange} errorMessage={errorMessage} />);
        return screen.getByRole('textbox', { name: 'Hotkey input' });
    };

    beforeEach(() => {
        mockOnHotkeyChange.mockClear();
    });

    describe('display', () => {
        it('renders empty with placeholder when hotkey is null', () => {
            const input = renderHotkeyField(null);

            expect(input).toHaveAttribute('placeholder', 'Hotkey');
            expect(input).toHaveValue('');
        });

        it('renders empty when hotkey is undefined', () => {
            const input = renderHotkeyField(undefined);

            expect(input).toHaveValue('');
        });

        it('displays formatted hotkey value in uppercase', () => {
            const input = renderHotkeyField('ctrl+s');

            expect(input).toHaveValue('CTRL+S');
        });

        it('shows error message when provided', () => {
            renderHotkeyField('a', 'That hotkey is already in use');

            expect(screen.getByText('That hotkey is already in use')).toBeInTheDocument();
        });
    });

    describe('key capture', () => {
        it.each([
            ['simple key', { key: 'a' }, 'a'],
            ['ctrl modifier', { key: 's', ctrlKey: true }, 'ctrl+s'],
            ['alt modifier', { key: 'a', altKey: true }, 'alt+a'],
            ['meta modifier', { key: 'a', metaKey: true }, 'meta+a'],
            ['shift modifier', { key: 's', shiftKey: true }, 'shift+s'],
            ['ctrl+shift', { key: 's', ctrlKey: true, shiftKey: true }, 'ctrl+shift+s'],
            [
                'all modifiers',
                { key: 'a', ctrlKey: true, metaKey: true, altKey: true, shiftKey: true },
                'ctrl+meta+alt+shift+a',
            ],
        ])('captures %s', (_, keyEvent, expected) => {
            const input = renderHotkeyField();

            fireEvent.keyDown(input, keyEvent);

            expect(mockOnHotkeyChange).toHaveBeenCalledWith(expected);
        });
    });

    describe('ignored keys', () => {
        it.each(['Control', 'Alt', 'Shift', 'Meta'])('ignores standalone %s key press', (key) => {
            const input = renderHotkeyField();

            fireEvent.keyDown(input, { key });

            expect(mockOnHotkeyChange).not.toHaveBeenCalled();
        });
    });

    describe('removing hotkeys', () => {
        it('"Backspace" key press should remove existing hotkeys', () => {
            const input = renderHotkeyField('ctrl+p');

            fireEvent.keyDown(input, { key: 'Backspace' });

            expect(mockOnHotkeyChange).toHaveBeenCalledWith(null);
        });
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { getMockedLabel } from '../../../../../mocks/mock-labels';
import { LabelRow, LabelRowProps } from './label-row.component';

describe('LabelRow', () => {
    const renderApp = ({
        label = getMockedLabel(),
        isSelected = false,
        isPinned = false,
        onSelect = rs.fn(),
        onDelete = rs.fn(),
        onTogglePin = rs.fn(),
        onUpdate = rs.fn(),
        validateName = rs.fn(() => undefined),
        validateHotkey = rs.fn(() => undefined),
    }: Partial<LabelRowProps> = {}) => {
        render(
            <LabelRow
                label={label}
                isSelected={isSelected}
                isPinned={isPinned}
                onSelect={onSelect}
                onDelete={onDelete}
                onTogglePin={onTogglePin}
                onUpdate={onUpdate}
                validateName={validateName}
                validateHotkey={validateHotkey}
            />
        );

        return { onUpdate, onSelect, onDelete, onTogglePin };
    };

    describe('handleUpdateName', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'old name', color: '#ffff00', hotkey: 'a' });

        it('calls onUpdate with trimmed name on blur', async () => {
            const onUpdate = rs.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, 'new name');
            await userEvent.tab();

            expect(onUpdate).toHaveBeenCalledWith(label.id, expect.objectContaining({ name: 'new name' }));
        });

        it('calls onUpdate on Enter key', async () => {
            const onUpdate = rs.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, 'new name{Enter}');

            expect(onUpdate).toHaveBeenCalledWith(label.id, expect.objectContaining({ name: 'new name' }));
        });

        it('does not call onUpdate when name has validation error', async () => {
            const onUpdate = rs.fn();
            const validateName = rs.fn(() => 'Name already exists');

            renderApp({ label, onUpdate, validateName });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, 'duplicate');
            await userEvent.tab();

            expect(onUpdate).not.toHaveBeenCalled();
        });

        it('does not call onUpdate when name is empty', async () => {
            const onUpdate = rs.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.tab();

            expect(onUpdate).not.toHaveBeenCalled();
        });

        it('does not call onUpdate when name has not changed', async () => {
            const onUpdate = rs.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, label.name);
            await userEvent.tab();

            expect(onUpdate).not.toHaveBeenCalled();
        });
    });

    describe('handleHotkeyChange', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'old name', color: '#ffff00', hotkey: 'a' });

        it('calls onUpdate on Enter when hotkey is valid and is changed', async () => {
            const onUpdate = rs.fn();

            renderApp({ label, onUpdate });

            const hotkeyInput = screen.getByLabelText(/Hotkey input/i);
            await userEvent.click(hotkeyInput);
            await userEvent.keyboard('b');
            await userEvent.keyboard('{Enter}');

            expect(onUpdate).toHaveBeenCalledWith(label.id, expect.objectContaining({ hotkey: 'b' }));
        });

        it('does not call onUpdate on Enter when hotkey is valid but is not changed', async () => {
            const onUpdate = rs.fn();

            renderApp({ label, onUpdate });

            const hotkeyInput = screen.getByLabelText(/Hotkey input/i);
            await userEvent.click(hotkeyInput);
            await userEvent.keyboard('{Enter}');

            expect(onUpdate).not.toHaveBeenCalled();
        });

        it('calls onUpdate when input is no longer focused', async () => {
            const onUpdate = rs.fn();

            renderApp({ label, onUpdate });

            const hotkeyInput = screen.getByLabelText(/Hotkey input/i);
            await userEvent.click(hotkeyInput);
            await userEvent.keyboard('b');
            fireEvent.blur(hotkeyInput);

            expect(onUpdate).toHaveBeenCalledWith(label.id, expect.objectContaining({ hotkey: 'b' }));
        });

        it('does not call onUpdate when hotkey has validation error', async () => {
            const onUpdate = rs.fn();
            const validateHotkey = rs.fn(() => 'Hotkey already in use');

            renderApp({ label, onUpdate, validateHotkey });

            const hotkeyInput = screen.getByLabelText(/Hotkey input/i);
            await userEvent.click(hotkeyInput);
            await userEvent.keyboard('{Enter}');

            expect(onUpdate).not.toHaveBeenCalled();
        });

        it('sends undefined for hotkey when hotkey is empty', async () => {
            const onUpdate = rs.fn();
            const labelNoHotkey = getMockedLabel({ id: 'label-1', name: 'old name', color: '#ffff00', hotkey: '' });

            renderApp({ label: labelNoHotkey, onUpdate });

            const hotkeyInput = screen.getByLabelText(/Hotkey input/i);
            await userEvent.click(hotkeyInput);
            await userEvent.keyboard('{Enter}');

            expect(onUpdate).toHaveBeenCalledWith(labelNoHotkey.id, expect.objectContaining({ hotkey: undefined }));
        });
    });

    describe('name validation display', () => {
        it('displays validation error message for name', () => {
            const label = getMockedLabel({ name: 'duplicate' });
            const validateName = rs.fn(() => 'Name already exists');

            renderApp({ label, validateName });

            expect(screen.getByText('Name already exists')).toBeInTheDocument();
        });
    });

    describe('hotkey validation display', () => {
        it('displays validation error message for hotkey', () => {
            const label = getMockedLabel({ hotkey: 'a' });
            const validateHotkey = rs.fn(() => 'Hotkey already in use');

            renderApp({ label, validateHotkey });

            expect(screen.getByText('Hotkey already in use')).toBeInTheDocument();
        });
    });
});

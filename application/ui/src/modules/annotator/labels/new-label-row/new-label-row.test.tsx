// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import * as actualLabelUtils from '../../../../shared/label-utils' with { rstest: 'importActual' };
import { NewLabelRow } from './new-label-row.component';

rs.mock('../../../../shared/label-utils', () => ({
    ...actualLabelUtils,
    getRandomDistinctColor: () => '#123456',
}));

describe('NewLabelRow', () => {
    const getNameInput = () => screen.getByRole('textbox', { name: 'New label name' });
    const getHotkeyInput = () => screen.getByRole('textbox', { name: 'Hotkey input' });

    it('renders name input and row controls', () => {
        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={rs.fn()}
                onCancel={rs.fn()}
                validateName={rs.fn(() => undefined)}
            />
        );

        expect(getNameInput()).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Create new label' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Cancel new label' })).toBeInTheDocument();
    });

    it('disables create button for empty or invalid names and enables it for valid names', async () => {
        const user = userEvent.setup();
        const validateName = rs.fn((name: string) => (name === 'valid-label' ? undefined : 'invalid name'));

        render(
            <NewLabelRow validateHotkey={rs.fn()} onSave={rs.fn()} onCancel={rs.fn()} validateName={validateName} />
        );

        const nameInput = getNameInput();
        const createButton = screen.getByRole('button', { name: 'Create new label' });

        expect(createButton).toBeDisabled();

        await user.type(nameInput, 'invalid-label');
        expect(createButton).toBeDisabled();

        await user.clear(nameInput);
        await user.type(nameInput, '   ');
        expect(createButton).toBeDisabled();

        await user.clear(nameInput);
        await user.type(nameInput, ' valid-label ');
        expect(createButton).toBeEnabled();
    });

    it('calls onSave with trimmed name and current color when create is clicked', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={onSave}
                onCancel={rs.fn()}
                validateName={rs.fn(() => undefined)}
            />
        );

        await user.type(getNameInput(), '  New class  ');
        await user.type(getHotkeyInput(), 'D');
        fireEvent.click(screen.getByRole('button', { name: 'Create new label' }));

        expect(onSave).toHaveBeenCalledWith('New class', '#123456', 'D');
    });

    it('does not call onSave when create is clicked for invalid name', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={onSave}
                onCancel={rs.fn()}
                validateName={rs.fn(() => 'invalid name')}
            />
        );

        await user.type(getNameInput(), 'invalid-label');
        fireEvent.click(screen.getByRole('button', { name: 'Create new label' }));

        expect(onSave).not.toHaveBeenCalled();
    });

    it('saves when Enter is pressed', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={onSave}
                onCancel={rs.fn()}
                validateName={rs.fn(() => undefined)}
            />
        );

        const nameInput = getNameInput();
        await user.type(nameInput, '  Enter label  ');
        await user.type(getHotkeyInput(), 'D');

        fireEvent.keyDown(nameInput, { key: 'Enter' });

        expect(onSave).toHaveBeenCalledWith('Enter label', '#123456', 'D');
    });

    it('cancels when Escape is pressed', async () => {
        const user = userEvent.setup();
        const onCancel = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={rs.fn()}
                onCancel={onCancel}
                validateName={rs.fn(() => undefined)}
            />
        );

        const nameInput = getNameInput();
        await user.type(nameInput, '  Enter label  ');

        fireEvent.keyDown(nameInput, { key: 'Escape' });

        expect(onCancel).toHaveBeenCalled();
    });

    it('does nothing on blur when focus stays inside the row', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();
        const onCancel = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={onSave}
                onCancel={onCancel}
                validateName={rs.fn(() => undefined)}
            />
        );

        const nameInput = getNameInput();
        const cancelButton = screen.getByRole('button', { name: 'Cancel new label' });

        await user.type(nameInput, 'Inside row blur');
        fireEvent.blur(nameInput, { relatedTarget: cancelButton });

        expect(onSave).not.toHaveBeenCalled();
        expect(onCancel).not.toHaveBeenCalled();
    });

    it('saves on blur when focus leaves the row and name can be saved', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={onSave}
                onCancel={rs.fn()}
                validateName={rs.fn(() => undefined)}
            />
        );

        const nameInput = getNameInput();
        await user.type(nameInput, '  Blur save  ');
        fireEvent.blur(nameInput, { relatedTarget: null });

        expect(onSave).toHaveBeenCalledWith('Blur save', '#123456');
    });

    it('cancels on blur when focus leaves row and name is empty after trimming', async () => {
        const user = userEvent.setup();
        const onCancel = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn()}
                onSave={rs.fn()}
                onCancel={onCancel}
                validateName={rs.fn(() => undefined)}
            />
        );

        const nameInput = getNameInput();
        await user.type(nameInput, '   ');
        fireEvent.blur(nameInput, { relatedTarget: null });

        expect(onCancel).toHaveBeenCalled();
    });

    it('calls onSave when Enter is pressed in hotkey field with a valid hotkey', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn(() => undefined)}
                onSave={onSave}
                onCancel={rs.fn()}
                validateName={rs.fn(() => undefined)}
            />
        );

        await user.type(getNameInput(), 'MyLabel');
        const hotkeyInput = getHotkeyInput();
        await user.type(hotkeyInput, 'D');
        await user.type(hotkeyInput, '{Enter}');

        expect(onSave).toHaveBeenCalledWith('MyLabel', '#123456', 'D');
    });

    it('does not call onSave when Enter is pressed in hotkey field with an invalid hotkey', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn(() => 'That hotkey is already in use')}
                onSave={onSave}
                onCancel={rs.fn()}
                validateName={rs.fn(() => undefined)}
            />
        );

        await user.type(getNameInput(), 'MyLabel');
        const hotkeyInput = getHotkeyInput();
        await user.type(hotkeyInput, 'B');
        await user.type(hotkeyInput, '{Enter}');

        expect(onSave).not.toHaveBeenCalled();
    });

    it('calls onSave with undefined hotkey when Enter is pressed with empty hotkey', async () => {
        const user = userEvent.setup();
        const onSave = rs.fn();

        render(
            <NewLabelRow
                validateHotkey={rs.fn(() => undefined)}
                onSave={onSave}
                onCancel={rs.fn()}
                validateName={rs.fn(() => undefined)}
            />
        );

        await user.type(getNameInput(), 'MyLabel');
        const hotkeyInput = getHotkeyInput();
        await user.click(hotkeyInput);
        await user.type(hotkeyInput, '{Enter}');

        expect(onSave).toHaveBeenCalledWith('MyLabel', '#123456', undefined);
    });
});

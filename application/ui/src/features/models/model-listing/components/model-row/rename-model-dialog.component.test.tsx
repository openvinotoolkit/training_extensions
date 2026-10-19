// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { RenameModelDialog } from './rename-model-dialog.component';

describe('RenameModelDialog', () => {
    const defaultProps = {
        currentName: 'Current Model Name',
        onRename: rs.fn(),
        onClose: rs.fn(),
    };

    describe('basic rendering', () => {
        it('should render dialog with heading and current name in input', () => {
            render(<RenameModelDialog {...defaultProps} />);

            expect(screen.getByRole('heading', { name: 'Rename model' })).toBeInTheDocument();
            expect(screen.getByRole('textbox')).toHaveValue('Current Model Name');
            expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Rename' })).toBeInTheDocument();
        });

        it('should have Rename button disabled when name is same as current', () => {
            render(<RenameModelDialog {...defaultProps} />);

            expect(screen.getByRole('button', { name: 'Rename' })).toBeDisabled();
        });

        it('should enable Rename button when name is changed', async () => {
            render(<RenameModelDialog {...defaultProps} />);

            const input = screen.getByRole('textbox');
            await userEvent.clear(input);
            await userEvent.type(input, 'New Model Name');

            expect(screen.getByRole('button', { name: 'Rename' })).not.toBeDisabled();
        });
    });

    describe('user interactions', () => {
        it('should call onClose when Cancel button is clicked', async () => {
            render(<RenameModelDialog {...defaultProps} />);

            await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));

            expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
        });

        it('should call onRename with trimmed new name when form is submitted', async () => {
            render(<RenameModelDialog {...defaultProps} />);

            const input = screen.getByRole('textbox');
            await userEvent.clear(input);
            await userEvent.type(input, '  New Model Name  ');

            await userEvent.click(screen.getByRole('button', { name: 'Rename' }));

            expect(defaultProps.onRename).toHaveBeenCalledWith('New Model Name');
            expect(defaultProps.onRename).toHaveBeenCalledTimes(1);
        });

        it('should update input value as user types', async () => {
            render(<RenameModelDialog {...defaultProps} />);

            const input = screen.getByRole('textbox');
            await userEvent.clear(input);
            await userEvent.type(input, 'Updated Name');

            expect(input).toHaveValue('Updated Name');
        });
    });

    describe('pending state', () => {
        it('should show pending state on Rename button when isPending is true', () => {
            render(<RenameModelDialog {...defaultProps} currentName='Old Name' isPending={true} />);

            const input = screen.getByRole('textbox');
            expect(input).toHaveValue('Old Name');

            const renameButton = screen.getByRole('button', { name: /Rename/i });
            expect(renameButton).toHaveAttribute('aria-label', 'pending');
        });
    });

    describe('edge cases', () => {
        it('should handle empty currentName', () => {
            render(<RenameModelDialog {...defaultProps} currentName='' />);

            expect(screen.getByRole('textbox')).toHaveValue('');
        });

        it('should disable Rename button when trimmed new name matches current name', async () => {
            render(<RenameModelDialog {...defaultProps} currentName='Model Name' />);

            const input = screen.getByRole('textbox');
            await userEvent.clear(input);
            await userEvent.type(input, '  Model Name  ');

            expect(screen.getByRole('button', { name: 'Rename' })).toBeDisabled();
        });
    });
});

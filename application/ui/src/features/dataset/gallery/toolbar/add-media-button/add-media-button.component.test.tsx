// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { acceptedExtensions } from '../../../../../shared/media-file-utils';
import { AddMediaButton } from './add-media-button.component';

describe('AddMediaButton', () => {
    it('calls onFilesSelected correctly', () => {
        const mockOnFilesSelected = rs.fn();
        const mockFile = new File(['file content'], 'test-image.jpg', {
            type: 'image/jpeg',
            lastModified: Date.now(),
        });

        render(<AddMediaButton onFileUpload={mockOnFilesSelected} />);

        const input = screen.getByTestId('upload-media-input');
        fireEvent.change(input, { target: { files: [mockFile] } });

        expect(mockOnFilesSelected).toHaveBeenCalledWith([mockFile]);
    });

    it('sets the expected accepted file extensions', () => {
        render(<AddMediaButton onFileUpload={rs.fn()} />);

        const input = screen.getByTestId('upload-media-input');

        expect(input).toHaveAttribute('accept', acceptedExtensions);
    });

    it('opens file picker when button is clicked', () => {
        const mockOnFilesSelected = rs.fn();

        render(<AddMediaButton onFileUpload={mockOnFilesSelected} />);

        const button = screen.getByRole('button', { name: /Upload media/ });
        const input = screen.getByTestId('upload-media-input') as HTMLInputElement;

        const clickSpy = rs.spyOn(input, 'click');
        fireEvent.click(button);

        expect(clickSpy).toHaveBeenCalled();
    });

    it('disables button when isDisabled prop is true', () => {
        render(<AddMediaButton onFileUpload={rs.fn()} isDisabled />);

        const button = screen.getByRole('button', { name: /Upload media/ });

        expect(button).toBeDisabled();
    });
});

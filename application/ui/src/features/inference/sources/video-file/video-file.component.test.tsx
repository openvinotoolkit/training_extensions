// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent } from 'react';

import { Form } from '@geti-ui/ui';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { VideoFile } from './video-file.component';

describe('VideoFile', () => {
    const renderApp = (props: Parameters<typeof VideoFile>[0] = {}) => {
        const handleSubmit = vi.fn((event: FormEvent) => event.preventDefault());

        render(
            <Form validationBehavior='native' onSubmit={handleSubmit}>
                <VideoFile {...props} />
                <button type='submit'>Submit</button>
            </Form>
        );

        return { handleSubmit };
    };

    const uploadFile = async (name = 'sample.mp4') => {
        const file = new File(['fake-video-bytes'], name, { type: 'video/mp4' });
        const fileInput = screen.getByTestId('upload-video-file');
        await userEvent.upload(fileInput, file);

        return file;
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders the path input pre-filled from defaultState', () => {
        renderApp({
            defaultState: {
                id: '1',
                name: 'My source',
                source_type: 'video_file',
                video_path: '/a/b.mp4',
                loop: false,
            },
        });

        expect(screen.getByRole('textbox', { name: /Video file path/ })).toHaveValue('/a/b.mp4');
    });

    it('allows typing a path directly into the input', async () => {
        renderApp();

        const pathField = screen.getByRole('textbox', { name: /Video file path/ });
        await userEvent.type(pathField, '/a/b.mp4');

        expect(pathField).toHaveValue('/a/b.mp4');
    });

    it('marks the form invalid until a path is set', () => {
        renderApp();

        const pathField = screen.getByRole('textbox', { name: /Video file path/ }) as HTMLInputElement;
        expect(pathField.checkValidity()).toBe(false);
    });

    it('gives the hidden file input a `video_file` name so it is included in the native FormData', async () => {
        renderApp();

        const fileInput = screen.getByTestId('upload-video-file') as HTMLInputElement;

        await waitFor(() => {
            expect(fileInput).toHaveAttribute('name', 'video_file');
        });
    });

    it('selecting a file stores it locally without making any network request', async () => {
        const { handleSubmit } = renderApp();

        const file = await uploadFile();

        expect(screen.getByText('Selected: sample.mp4')).toBeVisible();

        const fileInput = screen.getByTestId('upload-video-file') as HTMLInputElement;
        expect(fileInput.files?.[0]).toBe(file);
        expect(handleSubmit).not.toHaveBeenCalled();
    });

    it('un-requires and clears the path input once a file is selected', async () => {
        renderApp({
            defaultState: {
                id: '1',
                name: 'My source',
                source_type: 'video_file',
                video_path: '/a/b.mp4',
                loop: false,
            },
        });

        await uploadFile();

        const pathField = screen.getByRole('textbox', { name: /Video file path/ }) as HTMLInputElement;
        expect(pathField).toBeEnabled();
        expect(pathField).toHaveValue('');
        expect(pathField.checkValidity()).toBe(true);
    });

    it('typing a path clears a previously selected file', async () => {
        renderApp();

        await uploadFile();
        expect(screen.getByText('Selected: sample.mp4')).toBeVisible();

        const pathField = screen.getByRole('textbox', { name: /Video file path/ });
        await userEvent.type(pathField, '/a/b.mp4');

        expect(screen.queryByText('Selected: sample.mp4')).not.toBeInTheDocument();
        expect(pathField).toHaveValue('/a/b.mp4');

        const fileInput = screen.getByTestId('upload-video-file') as HTMLInputElement;
        expect(fileInput.files?.length ?? 0).toBe(0);
    });

    it('selecting a file clears a previously typed path', async () => {
        renderApp();

        const pathField = screen.getByRole('textbox', { name: /Video file path/ });
        await userEvent.type(pathField, '/a/b.mp4');
        expect(pathField).toHaveValue('/a/b.mp4');

        await uploadFile();

        expect(pathField).toHaveValue('');
        expect(screen.getByText('Selected: sample.mp4')).toBeVisible();
    });

    it('allows submission right after selecting a file, without typing a path', async () => {
        const { handleSubmit } = renderApp();

        await uploadFile();

        await userEvent.click(screen.getByRole('button', { name: 'Submit' }));

        expect(handleSubmit).toHaveBeenCalled();
    });

    it('allows submission after typing a path, without selecting a file', async () => {
        const { handleSubmit } = renderApp();

        const pathField = screen.getByRole('textbox', { name: /Video file path/ });
        await userEvent.type(pathField, '/a/b.mp4');

        await userEvent.click(screen.getByRole('button', { name: 'Submit' }));

        expect(handleSubmit).toHaveBeenCalled();
    });
    it('shows the transfer progress and lets the user cancel it', async () => {
        const onCancelUpload = vi.fn();
        renderApp({ uploadProgress: { bytesSent: 500, bytesTotal: 2000 }, onCancelUpload });

        expect(screen.getByRole('progressbar', { name: 'Upload progress' })).toHaveAttribute('aria-valuenow', '25');
        expect(screen.getByText('500 B of 2 kB')).toBeVisible();

        await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));

        expect(onCancelUpload).toHaveBeenCalledOnce();
    });

    it('does not show any progress when no transfer is running', () => {
        renderApp({ uploadProgress: null });

        expect(screen.queryByRole('progressbar')).toBeNull();
    });

    it('hides the cancel button once the transfer has completed', () => {
        const onCancelUpload = vi.fn();
        renderApp({ uploadProgress: { bytesSent: 2000, bytesTotal: 2000 }, onCancelUpload });

        expect(screen.getByRole('progressbar', { name: 'Upload progress' })).toHaveAttribute('aria-valuenow', '100');
        expect(screen.queryByRole('button', { name: 'Cancel' })).toBeNull();
    });
});

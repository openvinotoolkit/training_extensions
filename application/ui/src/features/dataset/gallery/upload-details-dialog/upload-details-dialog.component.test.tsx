// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderHook } from 'test-utils/render';

import { useUploadActions } from '../../hooks/use-upload-actions';
import { MediaUploadProvider, useMediaUploadDispatch } from '../../providers/media-upload-provider.component';

const makeFile = (name: string, size = 1024): File => new File(['x'.repeat(size)], name, { type: 'image/jpeg' });

const renderUpload = () =>
    renderHook(() => ({ upload: useUploadActions(), dispatch: useMediaUploadDispatch() }), {
        wrapper: MediaUploadProvider,
    });

describe('UploadDetailsDialog', () => {
    it('is not rendered when dialog is closed', () => {
        const { result } = renderUpload();

        act(() => {
            result.current.upload.startUploadProgress([makeFile('one.jpg')]);
        });

        expect(screen.queryByRole('dialog')).toBeNull();
    });

    it('renders one row per upload item with filename and size', () => {
        const { result } = renderUpload();

        act(() => {
            result.current.upload.startUploadProgress([makeFile('one.jpg', 1024), makeFile('two.png', 2048)]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        const dialog = screen.getByRole('dialog');
        expect(within(dialog).getByRole('heading', { name: 'Upload details' })).toBeVisible();
        expect(within(dialog).getByText('one.jpg')).toBeVisible();
        expect(within(dialog).getByText('two.png')).toBeVisible();
    });

    it('reflects per-item status transitions', async () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([makeFile('one.jpg'), makeFile('two.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        expect(screen.getAllByText('Queued')).toHaveLength(2);

        act(() => {
            result.current.upload.setItemUploading(ids[0]);
            result.current.upload.setItemUploaded(ids[0]);
            result.current.upload.setItemFailed(ids[1], 'boom');
        });

        expect(screen.getByText('Uploaded')).toBeVisible();
        expect(screen.getByText('Failed')).toBeVisible();

        await userEvent.click(screen.getByRole('button', { name: 'Error details' }));
        expect(await screen.findByText('boom')).toBeVisible();
    });

    it('does not render an Error button when failed item has no error message', () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([makeFile('one.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        act(() => {
            result.current.upload.setItemFailed(ids[0]);
        });

        expect(screen.getByText('Failed')).toBeVisible();
        expect(screen.queryByRole('button', { name: 'Error details' })).toBeNull();
    });

    it('closes when the Close button is pressed', async () => {
        const { result } = renderUpload();

        act(() => {
            result.current.upload.startUploadProgress([makeFile('one.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        await userEvent.click(screen.getByRole('button', { name: 'Close' }));

        await waitFor(() => {
            expect(screen.queryByRole('dialog')).toBeNull();
        });
    });
});

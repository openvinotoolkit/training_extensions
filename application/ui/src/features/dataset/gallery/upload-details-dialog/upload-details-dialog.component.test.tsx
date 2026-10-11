// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderHook } from 'test-utils/render';

import { useUploadActions } from '../../hooks/use-upload-actions';
import { useMediaUploadDispatch } from '../../providers/media-upload-context';
import { MediaUploadProvider } from '../../providers/media-upload-provider.component';

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

    it('shows the singular in-progress subheader for a single queued item', () => {
        const { result } = renderUpload();

        act(() => {
            result.current.upload.startUploadProgress([makeFile('one.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        expect(screen.getByText('Uploading 1 item - 0 uploaded')).toBeVisible();
    });

    it('shows the plural in-progress subheader for multiple queued items', () => {
        const { result } = renderUpload();

        act(() => {
            result.current.upload.startUploadProgress([
                makeFile('one.jpg'),
                makeFile('two.jpg'),
                makeFile('three.jpg'),
            ]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        expect(screen.getByText('Uploading 3 items - 0 uploaded')).toBeVisible();
    });

    it('shows the in-progress subheader with failures while still uploading', () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([
                makeFile('one.jpg'),
                makeFile('two.jpg'),
                makeFile('three.jpg'),
            ]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        act(() => {
            result.current.upload.setItemUploaded(ids[0]);
            result.current.upload.setItemFailed(ids[1], 'bad');
        });

        expect(screen.getByText('Uploading 3 items - 1 uploaded, 1 failed')).toBeVisible();
    });

    it('shows the mixed final subheader with correct singular/plural once uploading finishes', () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([makeFile('one.jpg'), makeFile('two.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        act(() => {
            result.current.upload.setItemUploaded(ids[0]);
            result.current.upload.setItemFailed(ids[1], 'bad');
            result.current.upload.finishUploadProgress();
        });

        expect(screen.getByText('Uploaded 1 item, 1 failed')).toBeVisible();
    });
    it('shows a progress bar with the transferred bytes while an item is uploading', () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([makeFile('one.jpg', 4000)]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        act(() => {
            result.current.upload.setItemUploading(ids[0]);
            result.current.upload.setItemTransferProgress(ids[0], 1000);
        });

        expect(screen.getByRole('progressbar', { name: 'Upload progress' })).toHaveAttribute('aria-valuenow', '25');
        expect(screen.getByText('1 kB of 4 kB')).toBeVisible();
    });

    it('cancels a single upload from its row', async () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([makeFile('one.jpg'), makeFile('two.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });
        const signal = result.current.upload.getItemAbortSignal(ids[0]);

        await userEvent.click(screen.getByRole('button', { name: 'Cancel upload of one.jpg' }));

        expect(signal?.aborted).toBe(true);
        expect(result.current.upload.getItemAbortSignal(ids[1])?.aborted).toBe(false);
        expect(screen.getByText('Cancelled')).toBeVisible();
        expect(screen.queryByRole('button', { name: 'Cancel upload of one.jpg' })).toBeNull();
    });

    it('cancels all pending uploads but not the finished ones', async () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([makeFile('one.jpg'), makeFile('two.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });
        act(() => {
            result.current.upload.setItemUploaded(ids[0]);
        });

        await userEvent.click(screen.getByRole('button', { name: 'Cancel all' }));

        expect(screen.getByText('Uploaded')).toBeVisible();
        expect(screen.getByText('Cancelled')).toBeVisible();
        expect(screen.queryByRole('button', { name: 'Cancel all' })).toBeNull();
    });

    it('shows the cancelled final subheader when every upload was cancelled', () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([makeFile('one.jpg'), makeFile('two.jpg')]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        act(() => {
            result.current.upload.cancelItems(ids);
            result.current.upload.finishUploadProgress();
        });

        expect(screen.getByText('Cancelled 2 uploads')).toBeVisible();
    });

    it('shows a mixed final subheader including cancelled uploads when the batch was not fully cancelled', () => {
        const { result } = renderUpload();

        let ids: string[] = [];
        act(() => {
            ids = result.current.upload.startUploadProgress([
                makeFile('one.jpg'),
                makeFile('two.jpg'),
                makeFile('three.jpg'),
            ]);
            result.current.dispatch({ type: 'OPEN_DIALOG' });
        });

        act(() => {
            result.current.upload.setItemUploaded(ids[0]);
            result.current.upload.setItemFailed(ids[1], 'bad');
            result.current.upload.cancelItems([ids[2]]);
            result.current.upload.finishUploadProgress();
        });

        expect(screen.getByText('Uploaded 1 item, 1 failed, 1 cancelled')).toBeVisible();
    });
});

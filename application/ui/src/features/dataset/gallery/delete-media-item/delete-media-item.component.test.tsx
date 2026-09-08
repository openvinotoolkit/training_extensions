// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { DeleteMediaItem } from './delete-media-item.component';

describe('DeleteMediaItem', () => {
    it('deletes a single media item and shows a success toast', async () => {
        const itemId = '123';
        const mockedOnDeleted = vitest.fn();
        let requestBody: { media_ids?: string[] } | undefined;

        server.use(
            http.delete('/api/projects/{project_id}/dataset/media', async ({ request }) => {
                requestBody = (await request.json()) as { media_ids: string[] };
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<DeleteMediaItem itemsIds={[itemId]} onDeleted={mockedOnDeleted} />);

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        expect(await screen.findByText(/Are you sure you want to delete 1 item\?/i)).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`1 item deleted successfully`)).toBeVisible();
        expect(requestBody).toEqual({ media_ids: [itemId] });
        expect(mockedOnDeleted).toHaveBeenCalledWith([itemId]);
    });

    it('deletes multiple media items and shows a success toast', async () => {
        const itemsIds = ['123', '456', '789'];
        const mockedOnDeleted = vitest.fn();
        let requestBody: { media_ids?: string[] } | undefined;

        server.use(
            http.delete('/api/projects/{project_id}/dataset/media', async ({ request }) => {
                requestBody = (await request.json()) as { media_ids: string[] };
                return new HttpResponse(null, { status: 204 });
            })
        );

        render(<DeleteMediaItem itemsIds={itemsIds} onDeleted={mockedOnDeleted} />);

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        expect(await screen.findByText(/Are you sure you want to delete 3 items\?/i)).toBeVisible();

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`3 items deleted successfully`)).toBeVisible();
        expect(requestBody).toEqual({ media_ids: itemsIds });
        expect(mockedOnDeleted).toHaveBeenCalledWith(itemsIds);
    });

    it('shows an error toast when deleting media items fails', async () => {
        const itemsIds = ['123', '456'];
        const errorMessage = 'test error message';
        const mockedOnDeleted = vitest.fn();

        server.use(
            http.delete('/api/projects/{project_id}/dataset/media', () => {
                // @ts-expect-error error response schema
                return HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        render(<DeleteMediaItem itemsIds={itemsIds} onDeleted={mockedOnDeleted} />);

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(/Are you sure you want to delete 2 items\?/i);

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`Failed to delete, ${errorMessage}`)).toBeVisible();
        expect(mockedOnDeleted).not.toHaveBeenCalled();
    });

    describe('backspace hotkey', () => {
        it('opens the confirmation dialog when the hotkey is enabled', async () => {
            const itemsIds = ['123', '456'];

            render(<DeleteMediaItem itemsIds={itemsIds} isHotkeyEnabled />);

            fireEvent.keyDown(document, { key: 'Backspace', code: 'Backspace' });

            expect(await screen.findByText(/Are you sure you want to delete 2 items\?/i)).toBeVisible();
        });

        it('opens the confirmation dialog while a gallery item is focused', async () => {
            render(
                <>
                    <div role={'option'} aria-selected data-testid={'media-item'} tabIndex={0} />
                    <DeleteMediaItem itemsIds={['123']} isHotkeyEnabled />
                </>
            );

            fireEvent.keyDown(screen.getByTestId('media-item'), { key: 'Backspace', code: 'Backspace' });

            expect(await screen.findByText(/Are you sure you want to delete 1 item\?/i)).toBeVisible();
        });

        it('does not open the confirmation dialog when the hotkey is disabled', async () => {
            render(<DeleteMediaItem itemsIds={['123']} />);

            fireEvent.keyDown(document, { key: 'Backspace', code: 'Backspace' });

            await waitFor(() => {
                expect(screen.queryByText(/Are you sure you want to delete/i)).not.toBeInTheDocument();
            });
        });

        it('does not open the confirmation dialog when there are no selected items', async () => {
            render(<DeleteMediaItem itemsIds={[]} isHotkeyEnabled />);

            fireEvent.keyDown(document, { key: 'Backspace', code: 'Backspace' });

            await waitFor(() => {
                expect(screen.queryByText(/Are you sure you want to delete/i)).not.toBeInTheDocument();
            });
        });
    });
});

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ViewModes } from '@geti-ui/ui';
import { screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { MediaUploadProvider } from '../providers/media-upload-provider.component';
import { SelectedDataProvider, useSelectedData } from '../providers/selected-data-provider.component';
import { Gallery } from './gallery.component';

const uploadMediaMock = vi.fn();

let dropFiles: (files: File[]) => void = () => {};

vi.mock('../../../shared/drop-zone.utils', () => ({
    getFilesFromDropEvent: ({ files }: { files: File[] }) => Promise.resolve(files),
}));

vi.mock('@geti-ui/ui', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@geti-ui/ui')>();
    return {
        ...actual,
        // Virtualized items are laid out from measurements jsdom does not provide, so they never render
        Virtualizer: ({ children }: { children: ReactNode }) => <>{children}</>,
        AriaDropZone: ({
            children,
            onDrop,
        }: {
            children: ReactNode | ((state: { isDropTarget: boolean }) => ReactNode);
            onDrop?: (event: { files: File[] }) => void;
        }) => {
            dropFiles = (files) => onDrop?.({ files });
            return <>{typeof children === 'function' ? children({ isDropTarget: false }) : children}</>;
        },
    };
});

vi.mock('../api/use-media-upload', () => ({
    useMediaUpload: () => ({
        uploadMedia: uploadMediaMock,
        uploadProgress: { total: 0, completed: 0, succeeded: 0, failed: 0, isUploading: false },
    }),
}));

vi.mock('./hooks/use-select-dataset-item.hook', () => ({
    useSelectDatasetItem: () => ({
        selectedMediaItem: null,
        onSelectedMediaItemChange: vi.fn(),
    }),
}));

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

describe('Gallery drag-and-drop upload', () => {
    const renderGallery = async () => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            })
        );

        render(
            <MediaUploadProvider>
                <SelectedDataProvider>
                    <Gallery
                        items={[]}
                        viewMode={ViewModes.LARGE}
                        isPending={false}
                        hasActiveFilter={false}
                        isFetchingNextPage={false}
                        fetchNextPage={vi.fn()}
                        isMediaItemReviewedById={() => false}
                    />
                </SelectedDataProvider>
            </MediaUploadProvider>
        );

        await waitForElementToBeRemoved(() => screen.queryByRole('progressbar'));
    };

    beforeEach(() => {
        uploadMediaMock.mockReset();
    });

    it('uploads supported files without showing an error toast', async () => {
        await renderGallery();

        dropFiles([
            new File([''], 'photo.png', { type: 'image/png' }),
            new File([''], 'clip.mp4', { type: 'video/mp4' }),
        ]);

        await waitFor(() => expect(uploadMediaMock).toHaveBeenCalledTimes(1));
        expect(uploadMediaMock.mock.calls[0][0].map((f: File) => f.name)).toEqual(['photo.png', 'clip.mp4']);
        expect(screen.queryByLabelText('toast')).not.toBeInTheDocument();
    });
});

describe('Gallery item deletion and selection', () => {
    const item = getMockedMediaImage({ id: 'item-1' });

    const GalleryWithSelectionCount = ({ items }: { items: (typeof item)[] }) => {
        const { selectedKeys } = useSelectedData();
        const count = selectedKeys instanceof Set ? selectedKeys.size : 0;

        return (
            <>
                {count > 0 && <p>{count} selected</p>}
                <Gallery
                    items={items}
                    viewMode={ViewModes.LARGE}
                    isPending={false}
                    hasActiveFilter={false}
                    isFetchingNextPage={false}
                    fetchNextPage={vi.fn()}
                    isMediaItemReviewedById={() => false}
                />
            </>
        );
    };

    const renderGalleryWithItems = async (items: (typeof item)[]) => {
        server.use(
            http.get('/api/projects/{project_id}', () => HttpResponse.json(getMockedProject({ id: 'project-123' }))),
            http.delete('/api/projects/{project_id}/dataset/media', () => new HttpResponse(null, { status: 204 }))
        );

        render(
            <MediaUploadProvider>
                <SelectedDataProvider>
                    <GalleryWithSelectionCount items={items} />
                </SelectedDataProvider>
            </MediaUploadProvider>
        );

        await waitForElementToBeRemoved(() => screen.queryByRole('progressbar'));
    };

    it('does not show a selected count after deleting a non-selected item', async () => {
        const user = userEvent.setup();
        await renderGalleryWithItems([item]);

        expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Media actions' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Confirm' }));
        await screen.findByText('1 item deleted successfully');

        expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();
    });

    it('removes the deleted item from the selected count', async () => {
        const user = userEvent.setup();
        await renderGalleryWithItems([item]);

        await user.click(screen.getAllByRole('option')[0]);
        expect(screen.getByText('1 selected')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Media actions' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Confirm' }));
        await screen.findByText('1 item deleted successfully');

        expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();
    });

    describe('multi selection', () => {
        const items = ['item-1', 'item-2', 'item-3', 'item-4'].map((id) => getMockedMediaImage({ id }));

        const getItem = (index: number) => screen.getAllByRole('option')[index];

        const getCheckbox = (id: string) =>
            screen.getByRole('checkbox', { name: `Selection state of media item ${id}` });

        const getCheckboxContainer = (id: string) => {
            const container = getCheckbox(id).closest('[data-floating-container]');

            if (container === null) {
                throw new Error(`Could not find the container wrapping the checkbox of ${id}`);
            }

            return container;
        };

        const clickWithModifier = async (
            user: ReturnType<typeof userEvent.setup>,
            modifier: 'Shift' | 'Control',
            index: number
        ) => {
            await user.keyboard(`{${modifier}>}`);
            await user.click(getItem(index));
            await user.keyboard(`{/${modifier}}`);
        };

        it('selects an item by clicking anywhere on it', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems(items);

            await user.click(getItem(0));

            expect(screen.getByText('1 selected')).toBeInTheDocument();
            expect(getCheckbox('item-1')).toBeChecked();
        });

        it('replaces the selection when clicking another item without a modifier', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems(items);

            await user.click(getItem(0));
            await user.click(getItem(2));

            expect(screen.getByText('1 selected')).toBeInTheDocument();
            expect(getCheckbox('item-1')).not.toBeChecked();
            expect(getCheckbox('item-3')).toBeChecked();
        });

        it('deselects the item when clicking it again', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems(items);

            await user.click(getItem(0));
            await user.click(getItem(0));

            expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();
            expect(getCheckbox('item-1')).not.toBeChecked();
        });

        it('adds an item to the selection when clicking with the ctrl key', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems(items);

            await user.click(getItem(0));
            await clickWithModifier(user, 'Control', 2);

            expect(screen.getByText('2 selected')).toBeInTheDocument();
            expect(getCheckbox('item-1')).toBeChecked();
            expect(getCheckbox('item-3')).toBeChecked();
        });

        it('selects every item between the previously selected item and the shift clicked one', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems(items);

            await user.click(getItem(1));
            await clickWithModifier(user, 'Shift', 3);

            expect(screen.getByText('3 selected')).toBeInTheDocument();
            expect(getCheckbox('item-1')).not.toBeChecked();
            expect(getCheckbox('item-2')).toBeChecked();
            expect(getCheckbox('item-3')).toBeChecked();
            expect(getCheckbox('item-4')).toBeChecked();
        });

        it('selects the range when shift clicking an item above the previously selected one', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems(items);

            await user.click(getItem(2));
            await clickWithModifier(user, 'Shift', 0);

            expect(screen.getByText('3 selected')).toBeInTheDocument();
            expect(getCheckbox('item-4')).not.toBeChecked();
        });

        it('replaces the selection when clicking on the checkbox overlay of another item', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems(items);

            // The checkbox itself ignores pointer events, so a click lands on the item behind it
            await user.click(getCheckboxContainer('item-1'));
            await user.click(getCheckboxContainer('item-3'));

            expect(screen.getByText('1 selected')).toBeInTheDocument();
            expect(getCheckbox('item-1')).not.toBeChecked();
            expect(getCheckbox('item-3')).toBeChecked();
        });

        it('does not select an item when using its actions menu', async () => {
            const user = userEvent.setup();
            await renderGalleryWithItems([item]);

            await user.click(screen.getByRole('button', { name: 'Media actions' }));

            expect(await screen.findByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
            expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();
        });
    });
});

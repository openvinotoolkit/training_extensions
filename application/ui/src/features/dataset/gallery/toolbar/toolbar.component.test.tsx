// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '@/api/types';
import { ViewModes } from '@geti-ui/ui';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedDatasetStatistics } from 'mocks/mock-dataset-item';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { getMockedMediaImage, getMultipleMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { isImage } from '../../../../shared/media-item-utils';
import { MediaUploadProvider } from '../../providers/media-upload-provider.component';
import { useSelectedData } from '../../providers/selected-data-provider.component';
import { Toolbar } from './toolbar.component';

const uploadMediaMock = rs.fn();
const onSelectedMediaItemChangeMock = rs.fn();

rs.mock('../../api/use-media-upload', () => ({
    useMediaUpload: () => ({
        uploadMedia: uploadMediaMock,
    }),
}));

rs.mock('../hooks/use-select-dataset-item.hook', () => ({
    useSelectDatasetItem: () => ({
        onSelectedMediaItemChange: onSelectedMediaItemChangeMock,
        selectedMediaItem: null,
    }),
}));

rs.mock('../../import-export/import-export.component', () => ({
    ImportExport: () => <button>Export/Import</button>,
}));

rs.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

rs.mock('../../providers/selected-data-provider.component', () => ({
    useSelectedData: rs.fn(() => ({
        selectedKeys: new Set(),
        setSelectedKeys: rs.fn(),
        toggleSelectedKeys: rs.fn(),
        isSelected: rs.fn().mockReturnValue(false),
    })),
}));

describe('Toolbar', () => {
    const renderToolbar = async (items: Media[] = [], route: string = '/projects/123', allMedia: Media[] = items) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            }),
            http.get('/api/projects/{project_id}/dataset/statistics', () => {
                return HttpResponse.json(
                    getMockedDatasetStatistics({
                        media_counts: {
                            images: items.filter(isImage).length,
                            videos: items.filter((item) => !isImage(item)).length,
                            video_frames: 0,
                        },
                    })
                );
            }),
            http.get('/api/projects/{project_id}/dataset/media', () => {
                return HttpResponse.json({
                    items: [getMockedMediaImage({})],
                    pagination: { offset: 0, limit: 1, count: 1, total: allMedia.length },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/media/ids', () => {
                return HttpResponse.json({ items: allMedia.map(({ id, type }) => ({ id, type })) });
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    pagination: {
                        total: items.length,
                        offset: 0,
                        limit: 0,
                        count: items.length,
                    },
                    items: [],
                });
            }),
            http.get('/api/projects/{project_id}/dataset/views', () => {
                return HttpResponse.json([getMockedDatasetView({ id: 'collection-one', name: 'Collection One' })]);
            })
        );

        const result = render(
            <MediaUploadProvider>
                <Toolbar
                    items={items}
                    viewMode={ViewModes.LARGE}
                    setViewMode={rs.fn()}
                    trainModel={<button>Train model</button>}
                />
            </MediaUploadProvider>,
            {
                route,
                path: '/projects/:projectId',
            }
        );

        await screen.findByRole('button', { name: 'Select dataset view' });

        return result;
    };

    beforeEach(() => {
        uploadMediaMock.mockClear();
        onSelectedMediaItemChangeMock.mockClear();
        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set(),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });
    });

    it('delegates selected files to useMediaUpload', async () => {
        const file = new File(['file-content'], 'media-item.jpg', { type: 'image/jpeg' });

        await renderToolbar();

        const input = screen.getByTestId('upload-media-input');
        fireEvent.change(input, { target: { files: [file] } });

        expect(uploadMediaMock).toHaveBeenCalledWith([file]);
    });

    it('disables annotate button when there are no items', async () => {
        await renderToolbar();

        expect(screen.getByRole('button', { name: 'Annotate' })).toBeDisabled();
    });

    it('calls onSelectedMediaItemChange with first item when annotate is clicked', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });
        const secondItem = getMockedMediaImage({ id: 'second-item' });

        await renderToolbar([firstItem, secondItem]);

        fireEvent.click(screen.getByRole('button', { name: 'Annotate' }));

        expect(onSelectedMediaItemChangeMock).toHaveBeenCalledWith(firstItem);
    });

    it('shows selected count and delete button when items are selected', async () => {
        const item1 = getMockedMediaImage({ id: '1' });
        const item2 = getMockedMediaImage({ id: '2' });

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([item1.id, item2.id]),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([item1, item2]);

        expect(await screen.findByText('2 selected')).toBeVisible();
        expect(screen.getByLabelText(/delete media item/i)).toBeVisible();
    });

    it('opens dataset statistics modal when clicking the statistics button', async () => {
        renderToolbar();

        const statsButton = await screen.findByLabelText('dataset statistics');
        fireEvent.click(statsButton);

        await waitFor(() => {
            expect(screen.getByText('Dataset Statistics')).toBeVisible();
            expect(screen.getByText('Number of media')).toBeVisible();
            expect(screen.getByText('Annotated images')).toBeVisible();
        });
    });

    it('hides "Export/Import", "Train model", "Annotate" buttons when at least one media is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(screen.getByRole('button', { name: 'Upload media' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Annotate' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Train model' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Export/Import' })).not.toBeInTheDocument();
    });

    it('shows "Export/Import", "Train model", "Annotate" buttons when no media is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set(),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(screen.getByRole('button', { name: 'Upload media' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Annotate' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Train model' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Export/Import' })).toBeInTheDocument();
    });

    it('hides filter and view controls when at least one media is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set(firstItem.id),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(screen.queryByRole('button', { name: 'dataset statistics' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /media status/i })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Filter by labels' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'More filters' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'View mode' })).not.toBeInTheDocument();
    });

    it('shows filter and view controls when no media is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        await renderToolbar([firstItem]);

        expect(await screen.findByRole('button', { name: 'dataset statistics' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Filter by labels' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'More filters' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /media status/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'View mode' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Sort Media by:/ })).toBeInTheDocument();
    });

    it('sorts by upload date, newest first, by default', async () => {
        await renderToolbar();

        expect(screen.getByRole('button', { name: /Sort Media by:/ })).toHaveTextContent('Upload date (newest)');
    });

    it('switches between newest and oldest upload date sorting', async () => {
        await renderToolbar();

        fireEvent.click(screen.getByRole('button', { name: /Sort Media by:/ }));
        fireEvent.click(await screen.findByRole('option', { name: 'Upload date (oldest)' }));

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Sort Media by:/ })).toHaveTextContent('Upload date (oldest)');
        });

        fireEvent.click(screen.getByRole('button', { name: /Sort Media by:/ }));
        fireEvent.click(await screen.findByRole('option', { name: 'Upload date (newest)' }));

        await waitFor(() => {
            expect(screen.getByRole('button', { name: /Sort Media by:/ })).toHaveTextContent('Upload date (newest)');
        });
    });

    it('hides sort picker when at least one media item is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(screen.queryByRole('button', { name: /Sort Media by:/ })).not.toBeInTheDocument();
    });

    it('does not display dataset view actions until media is selected', async () => {
        await renderToolbar();

        expect(screen.getByRole('button', { name: 'Annotate' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Save view' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Assign to existing view' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Unassign from this view' })).not.toBeInTheDocument();
    });

    it('displays save or assign when at least one media is selected on the entire dataset', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(await screen.findByText('1 selected')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Save view' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Assign to existing view' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Unassign from this view' })).not.toBeInTheDocument();
    });

    it('displays unassign media when at least one media is selected on a dataset view', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem], '/projects/123?datasetViewId=collection-one');

        expect(await screen.findByText('1 selected')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Unassign from this view' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Save view' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Assign to existing view' })).not.toBeInTheDocument();
    });

    it('always shows the dataset views selector, with or without a selection', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        const { unmount } = await renderToolbar([firstItem]);
        expect(screen.getByRole('button', { name: 'Select dataset view' })).toBeInTheDocument();
        unmount();

        rs.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: rs.fn(),
            toggleSelectedKeys: rs.fn(),
            isSelected: rs.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);
        expect(screen.getByRole('button', { name: 'Select dataset view' })).toBeInTheDocument();
    });

    describe('select all', () => {
        it('selects every media item matching the filters, not just the loaded pages', async () => {
            const setSelectedKeys = rs.fn();
            const allMedia = getMultipleMockedMediaImage(120);
            const loadedItems = allMedia.slice(0, 40);

            rs.mocked(useSelectedData).mockReturnValue({
                selectedKeys: new Set(),
                setSelectedKeys,
                toggleSelectedKeys: rs.fn(),
                isSelected: rs.fn().mockReturnValue(false),
            });

            await renderToolbar(loadedItems, '/projects/123', allMedia);

            const selectAll = screen.getByLabelText('select all');

            await waitFor(() => {
                expect(selectAll).toBeEnabled();
            });

            fireEvent.click(selectAll);

            await waitFor(() => {
                expect(setSelectedKeys).toHaveBeenCalledWith(new Set(allMedia.map(({ id }) => id)));
            });
        });

        it('clears the selection when everything is already selected', async () => {
            const setSelectedKeys = rs.fn();
            const allMedia = getMultipleMockedMediaImage(3);

            rs.mocked(useSelectedData).mockReturnValue({
                selectedKeys: new Set(allMedia.map(({ id }) => id)),
                setSelectedKeys,
                toggleSelectedKeys: rs.fn(),
                isSelected: rs.fn().mockReturnValue(true),
            });

            await renderToolbar(allMedia);

            const selectAll = screen.getByLabelText('select all');

            await waitFor(() => {
                expect(selectAll).toBeChecked();
            });

            fireEvent.click(selectAll);

            await waitFor(() => {
                expect(setSelectedKeys).toHaveBeenCalledWith(new Set());
            });
        });
    });
});

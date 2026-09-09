// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '@/api/types';
import { ViewModes } from '@geti-ui/ui';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedDatasetStatistics } from 'mocks/mock-dataset-item';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../api/utils';
import { server } from '../../../../msw-node-setup';
import { isImage } from '../../../../shared/media-item-utils';
import { useSelectedData } from '../../providers/selected-data-provider.component';
import { Toolbar } from './toolbar.component';

const uploadMediaMock = vi.fn();
const onSelectedMediaItemChangeMock = vi.fn();

vi.mock('../../api/use-media-upload', () => ({
    useMediaUpload: () => ({
        uploadMedia: uploadMediaMock,
        uploadProgress: {
            total: 0,
            completed: 0,
            succeeded: 0,
            failed: 0,
            isUploading: false,
        },
    }),
}));

vi.mock('../hooks/use-select-dataset-item.hook', () => ({
    useSelectDatasetItem: () => ({
        onSelectedMediaItemChange: onSelectedMediaItemChangeMock,
        selectedMediaItem: null,
    }),
}));

vi.mock('../../../models/train-model/train-model.component', () => ({
    TrainModel: () => <button>Train model</button>,
}));

vi.mock('../../import-export/import-export.component', () => ({
    ImportExport: () => <button>Export/Import</button>,
}));

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

vi.mock('../../providers/selected-data-provider.component', () => ({
    useSelectedData: vi.fn(() => ({
        selectedKeys: new Set(),
        setSelectedKeys: vi.fn(),
        toggleSelectedKeys: vi.fn(),
        isSelected: vi.fn().mockReturnValue(false),
    })),
}));

describe('Toolbar', () => {
    const renderToolbar = async (items: Media[] = [], route: string = '/projects/123') => {
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
                    pagination: { offset: 0, limit: 1, count: items.length, total: items.length },
                });
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

        const result = render(<Toolbar items={items} viewMode={ViewModes.LARGE} setViewMode={vi.fn()} />, {
            route,
            path: '/projects/:projectId',
        });

        await screen.findByRole('button', { name: 'Select dataset view' });

        return result;
    };

    beforeEach(() => {
        uploadMediaMock.mockClear();
        onSelectedMediaItemChangeMock.mockClear();
        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set(),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
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

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([item1.id, item2.id]),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
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

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(screen.getByRole('button', { name: 'Upload media' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Annotate' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Train model' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Export/Import' })).not.toBeInTheDocument();
    });

    it('shows "Export/Import", "Train model", "Annotate" buttons when no media is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set(),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(screen.getByRole('button', { name: 'Upload media' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Annotate' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Train model' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Export/Import' })).toBeInTheDocument();
    });

    it('hides filter and view controls when at least one media is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set(firstItem.id),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
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
        expect(screen.getByRole('button', { name: 'Newest first' })).toBeInTheDocument();
    });

    it('shows "Newest first" sort button by default', async () => {
        await renderToolbar();

        const sortButton = screen.getByRole('button', { name: 'Newest first' });

        expect(sortButton).toHaveTextContent('Newest first');
    });

    it('toggles the sort button label between "Newest first" and "Oldest first" when clicked', async () => {
        await renderToolbar();

        const newestFirstButton = screen.getByRole('button', { name: 'Newest first' });
        fireEvent.click(newestFirstButton);

        const oldestFirstButton = await screen.findByRole('button', { name: 'Oldest first' });
        expect(oldestFirstButton).toHaveTextContent('Oldest first');

        fireEvent.click(oldestFirstButton);

        const newestFirstButton2 = await screen.findByRole('button', { name: 'Newest first' });
        expect(newestFirstButton2).toHaveTextContent('Newest first');
    });

    it('hides sort button when at least one media item is selected', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(screen.queryByRole('button', { name: 'Newest first' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Oldest first' })).not.toBeInTheDocument();
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

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);

        expect(await screen.findByText('1 selected')).toBeVisible();
        expect(screen.getByRole('button', { name: 'Save view' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Assign to existing view' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Unassign from this view' })).not.toBeInTheDocument();
    });

    it('displays unassign media when at least one media is selected on a dataset view', async () => {
        const firstItem = getMockedMediaImage({ id: 'first-item' });

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
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

        vi.mocked(useSelectedData).mockReturnValue({
            selectedKeys: new Set([firstItem.id]),
            setSelectedKeys: vi.fn(),
            toggleSelectedKeys: vi.fn(),
            isSelected: vi.fn().mockReturnValue(false),
        });

        await renderToolbar([firstItem]);
        expect(screen.getByRole('button', { name: 'Select dataset view' })).toBeInTheDocument();
    });
});

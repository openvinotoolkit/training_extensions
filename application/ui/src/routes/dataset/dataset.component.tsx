// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Grid, View } from '@geti-ui/ui';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';
import { useViewMode } from 'hooks/use-view-mode.hook';

import {
    ActiveFilters,
    useHasActiveFilters,
} from '../../features/dataset/gallery/active-filters/active-filters.component';
import { Gallery } from '../../features/dataset/gallery/gallery.component';
import { Toolbar } from '../../features/dataset/gallery/toolbar/toolbar.component';
import { ExportJobsList } from '../../features/dataset/import-export/export-jobs-list/export-jobs-list.component';
import { ImportJobsList } from '../../features/dataset/import-export/import-jobs-list/import-jobs-list.component';
import { MediaUploadProvider } from '../../features/dataset/providers/media-upload-provider.component';
import { GalleryViewMode } from '../../shared/gallery-view-modes';

export const Dataset = () => {
    const [viewMode, setViewMode] = useViewMode('dataset-gallery-view-mode');

    const { items, isPending, isFetchingNextPage, fetchNextPage, isMediaItemReviewedById } =
        useDatasetMediaWithReviewStatus();

    const hasActiveFilter = useHasActiveFilters();

    return (
        <MediaUploadProvider>
            <Grid
                height='100%'
                gridArea='content'
                rows={['auto', 'auto', 'auto', 'minmax(0, 1fr)']}
                UNSAFE_style={{ padding: dimensionValue('size-300') }}
            >
                <View gridRow='1 / 2'>
                    <ExportJobsList predicate={({ datasetId }) => datasetId === null} />
                    <ImportJobsList />
                </View>

                <View gridRow='2 / 3'>
                    <Toolbar items={items} viewMode={viewMode} setViewMode={setViewMode} />
                </View>

                <View gridRow='3 / 4' marginBottom={hasActiveFilter ? 'size-200' : undefined}>
                    <ActiveFilters />
                </View>

                <View gridRow='4 / 5'>
                    <Gallery
                        items={items}
                        viewMode={viewMode as GalleryViewMode}
                        isPending={isPending}
                        fetchNextPage={fetchNextPage}
                        isMediaItemReviewedById={isMediaItemReviewedById}
                        hasActiveFilter={hasActiveFilter}
                        isFetchingNextPage={isFetchingNextPage}
                    />
                </View>
            </Grid>
        </MediaUploadProvider>
    );
};

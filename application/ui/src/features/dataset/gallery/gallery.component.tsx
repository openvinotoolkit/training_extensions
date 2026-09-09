// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '@/api/types';
import { MediaItem } from '@/components/media-item/media-item.component';
import { MediaThumbnail } from '@/components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '@/components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import { Checkbox, DialogContainer, dimensionValue, Flex, Selection, Size, ViewModes } from '@geti-ui/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty, isEqual } from 'lodash-es';
import { GridLayoutOptions } from 'react-aria-components';

import { type GalleryViewMode } from '../../../shared/gallery-view-modes';
import { getMediaDownloadUrl, getThumbnailUrl } from '../../../shared/media-url.utils';
import { MediaPreview } from '../media-preview/media-preview.component';
import { useSelectedData } from '../providers/selected-data-provider.component';
import { AnnotationStatusIcon } from './annotation-state-icon.component';
import { BulkLabelsAssignmentDialog } from './bulk-labels-assignment/bulk-labels-assignment-dialog.component';
import { DatasetDropZone } from './drop-zone.component';
import { EmptyDataset } from './empty-dataset.component';
import { useSelectDatasetItem } from './hooks/use-select-dataset-item.hook';
import { MediaItemActions } from './media-item-actions/media-item-actions.component';
import { MediaItemContextualHelp } from './media-item-contextual-help/media-item-contextual-help.component';
import { useUploadFiles } from './use-upload-files';

type GalleryProps = {
    items: Media[];
    viewMode: GalleryViewMode;
    isPending: boolean;
    hasActiveFilter: boolean;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
};

const VIEW_MODE_SETTINGS: Record<GalleryViewMode, GridLayoutOptions> = {
    [ViewModes.LARGE]: { minItemSize: new Size(300, 300), minSpace: new Size(10, 10), preserveAspectRatio: true },
    [ViewModes.MEDIUM]: { minItemSize: new Size(200, 200), minSpace: new Size(6, 6), preserveAspectRatio: true },
    [ViewModes.SMALL]: { minItemSize: new Size(120, 120), minSpace: new Size(4, 4), preserveAspectRatio: true },
};

type GalleryListProps = {
    items: Media[];
    viewMode: GalleryViewMode;
    isPending: boolean;
    isFetchingNextPage: boolean;
    fetchNextPage: () => void;
    isMediaItemReviewedById: (mediaItemId: string) => boolean;
    onSelectedMediaItemChange: (item: Media) => void;
};

const GalleryList = ({
    items,
    viewMode,
    isPending,
    isFetchingNextPage,
    fetchNextPage,
    onSelectedMediaItemChange,
    isMediaItemReviewedById,
}: GalleryListProps) => {
    const projectId = useProjectIdentifier();
    const { selectedKeys, setSelectedKeys, toggleSelectedKeys, isSelected } = useSelectedData();

    const handleSelectionChange = (keys: Selection) => {
        setSelectedKeys((previousKeys) => {
            const isSameItem = keys !== 'all' && keys.size === 1 && isEqual(previousKeys, keys);

            return isSameItem ? new Set() : keys;
        });
    };

    return (
        <VirtualizerGridLayout
            items={items}
            ariaLabel='data-collection-grid'
            selectionMode='multiple'
            selectionBehavior='replace'
            allowDuplicateSelectionEvents
            selectOnFocus={false}
            selectedKeys={selectedKeys}
            onSelectionChange={handleSelectionChange}
            layoutOptions={VIEW_MODE_SETTINGS[viewMode]}
            isPending={isPending}
            isLoadingMore={isFetchingNextPage}
            onLoadMore={fetchNextPage}
            contentItem={(item) => {
                const mediaUrl = getThumbnailUrl(projectId, item.id);
                const downloadUrl = getMediaDownloadUrl(projectId, item.id);
                const mediaFileName = `${item.name}.${item.format}`;
                const selected = isSelected(item.id);

                return (
                    <MediaItem
                        contentElement={() => (
                            <MediaThumbnail
                                item={item}
                                alt={item.name}
                                url={mediaUrl}
                                onDoubleClick={() => onSelectedMediaItemChange(item)}
                            />
                        )}
                        topLeftElement={() => (
                            <Flex
                                width={'size-200'}
                                height={'size-200'}
                                alignItems={'center'}
                                justifyContent={'center'}
                                // Set pointerEvents to 'none' to allow clicks to pass through
                                // to the MediaItemActions component
                                UNSAFE_style={{ margin: dimensionValue('size-150'), pointerEvents: 'none' }}
                            >
                                <Checkbox
                                    aria-label={`Selection state of media item ${item.id}`}
                                    isSelected={selected}
                                    isReadOnly
                                />
                            </Flex>
                        )}
                        topRightElement={() => (
                            <Flex alignItems={'center'} gap={'size-50'}>
                                <MediaItemContextualHelp item={item} />

                                <MediaItemActions
                                    id={item.id}
                                    onDeleted={selected ? toggleSelectedKeys : undefined}
                                    mediaUrl={downloadUrl}
                                    mediaFileName={mediaFileName}
                                    onAnnotate={() => onSelectedMediaItemChange(item)}
                                />
                            </Flex>
                        )}
                        bottomRightElement={() => (
                            <AnnotationStatusIcon isReviewed={isMediaItemReviewedById(item.id)} />
                        )}
                    />
                );
            }}
        />
    );
};

export const Gallery = ({
    items,
    viewMode,
    isPending,
    hasActiveFilter,
    isFetchingNextPage,
    fetchNextPage,
    isMediaItemReviewedById,
}: GalleryProps) => {
    const { selectedMediaItem, onSelectedMediaItemChange } = useSelectDatasetItem();

    const { isClassification, uploadFiles, clearFilesForLabelAssignment, filesForLabelAssignment } = useUploadFiles();

    const content =
        !isPending && isEmpty(items) ? (
            <EmptyDataset hasActiveFilter={hasActiveFilter} />
        ) : (
            <GalleryList
                items={items}
                viewMode={viewMode}
                isPending={isPending}
                fetchNextPage={fetchNextPage}
                isMediaItemReviewedById={isMediaItemReviewedById}
                onSelectedMediaItemChange={onSelectedMediaItemChange}
                isFetchingNextPage={isFetchingNextPage}
            />
        );

    return (
        <>
            <DatasetDropZone onFilesDropped={uploadFiles}>
                {content}

                <DialogContainer type={'fullscreenTakeover'} onDismiss={() => onSelectedMediaItemChange(null)}>
                    {selectedMediaItem !== null && (
                        <MediaPreview
                            close={() => onSelectedMediaItemChange(null)}
                            onSelectedMediaItem={onSelectedMediaItemChange}
                        />
                    )}
                </DialogContainer>
            </DatasetDropZone>

            {isClassification && (
                <BulkLabelsAssignmentDialog files={filesForLabelAssignment} onClose={clearFilesForLabelAssignment} />
            )}
        </>
    );
};

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useMemo, useState } from 'react';

import type { DatasetRevisionItem } from '@/api/types';
import { MediaItem } from '@/components/media-item/media-item.component';
import { MediaThumbnail } from '@/components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '@/components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import { DialogContainer, Flex, Loading, Size, Text, View, ViewModes } from '@geti-ui/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { GridLayoutOptions } from 'react-aria-components';

import { type GalleryViewMode } from '../../../../shared/gallery-view-modes';
import { getDatasetRevisionThumbnailUrl } from '../../../../shared/media-url.utils';
import { usePrefetchMediaItem } from '../../../annotator/hooks/use-prefetch-media-item.hook';
import { type SelectableModel } from '../../utils';
import { SubsetMediaDialog } from './subset-media-dialog.component';
import { datasetRevisionItemToMedia } from './utils';

const VIEW_MODE_SETTINGS: Record<GalleryViewMode, GridLayoutOptions> = {
    [ViewModes.LARGE]: { minItemSize: new Size(180, 180), minSpace: new Size(6, 6), preserveAspectRatio: true },
    [ViewModes.MEDIUM]: { minItemSize: new Size(120, 120), minSpace: new Size(4, 4), preserveAspectRatio: true },
    [ViewModes.SMALL]: { minItemSize: new Size(80, 80), minSpace: new Size(4, 4), preserveAspectRatio: true },
};

// How many loaded items must remain ahead of the newly selected one before the next page is requested
const LOAD_AHEAD_BUFFER = 1;

type SubsetGalleryProps = {
    items: DatasetRevisionItem[];
    datasetRevisionId: string;
    viewMode: GalleryViewMode;
    fetchNextPage: () => void;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
    isPending: boolean;
    selectedModel: SelectableModel | undefined;
};

const useSubsetNavigation = ({
    items,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
}: Pick<SubsetGalleryProps, 'items' | 'hasNextPage' | 'isFetchingNextPage' | 'fetchNextPage'>) => {
    const [selectedItemId, setSelectedItemId] = useState<string | null>(null);

    const selectedItemIndex = items.findIndex(({ id }) => id === selectedItemId);
    const selectedItem = selectedItemIndex === -1 ? null : items[selectedItemIndex];
    const nextItem = selectedItemIndex === -1 ? undefined : items[selectedItemIndex + 1];

    const nextMediaItem = useMemo(
        () => (nextItem === undefined ? undefined : datasetRevisionItemToMedia(nextItem)),
        [nextItem]
    );

    usePrefetchMediaItem(nextMediaItem);

    const selectPreviousItem =
        selectedItemIndex > 0 ? () => setSelectedItemId(items[selectedItemIndex - 1].id) : undefined;

    const selectNextItem =
        nextItem === undefined
            ? undefined
            : () => {
                  setSelectedItemId(nextItem.id);

                  const remainingItems = items.length - 1 - (selectedItemIndex + 1);

                  // Keep loading ahead so the user can keep walking through the subset
                  if (hasNextPage && !isFetchingNextPage && remainingItems < LOAD_AHEAD_BUFFER) {
                      fetchNextPage();
                  }
              };

    return {
        selectedItem,
        selectItem: (id: string) => setSelectedItemId(id),
        clearSelection: () => setSelectedItemId(null),
        selectPreviousItem,
        selectNextItem,
    };
};

export const SubsetGallery = ({
    items,
    viewMode,
    datasetRevisionId,
    hasNextPage,
    isFetchingNextPage,
    isPending,
    fetchNextPage,
    selectedModel,
}: SubsetGalleryProps) => {
    const projectId = useProjectIdentifier();
    const { selectedItem, selectItem, clearSelection, selectPreviousItem, selectNextItem } = useSubsetNavigation({
        items,
        hasNextPage,
        isFetchingNextPage,
        fetchNextPage,
    });

    if (isPending) {
        return (
            <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
                <Loading mode='inline' />
            </Flex>
        );
    }

    if (items.length === 0) {
        return (
            <Flex height={'100%'} alignItems={'center'} justifyContent={'center'}>
                <Text>No items in this subset</Text>
            </Flex>
        );
    }

    return (
        <>
            <View height={'size-5000'} width={'100%'}>
                <VirtualizerGridLayout
                    items={items}
                    ariaLabel={'subset media grid'}
                    selectionMode='none'
                    layoutOptions={VIEW_MODE_SETTINGS[viewMode]}
                    isLoadingMore={isFetchingNextPage}
                    onLoadMore={() => hasNextPage && fetchNextPage()}
                    contentItem={(item) => (
                        <MediaItem
                            contentElement={() => (
                                <MediaThumbnail
                                    // TODO: Revisit this once API supports required props in DatasetRevisionItem
                                    item={{ ...item, type: 'image' }}
                                    alt={`${item.subset} item`}
                                    url={getDatasetRevisionThumbnailUrl(projectId, datasetRevisionId, item.id)}
                                    onDoubleClick={() => selectItem(item.id)}
                                />
                            )}
                        />
                    )}
                />
            </View>

            <DialogContainer type={'fullscreen'} onDismiss={clearSelection}>
                {selectedItem && (
                    <Suspense fallback={<Loading />}>
                        <SubsetMediaDialog
                            item={selectedItem}
                            onClose={clearSelection}
                            selectedModel={selectedModel}
                            onSelectPreviousMediaItem={selectPreviousItem}
                            onSelectNextMediaItem={selectNextItem}
                        />
                    </Suspense>
                )}
            </DialogContainer>
        </>
    );
};

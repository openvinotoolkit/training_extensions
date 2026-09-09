// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import type { Media } from '@/api/types';
import { VirtualizerGridLayout } from '@/components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import { Flex, Size, useUnwrapDOMRef, View } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { SIDEBAR_MEDIA_SIZE } from '../constants';
import { Toolbar } from '../toolbar-container/toolbar-container.component';
import { SidebarMediaFilter } from './sidebar-media-filtering.component';
import { SidebarMediaItem } from './sidebar-media-item.component';
import { useKeyboardNavigation } from './use-keyboard-navigation.hook';

const layoutOptions = {
    maxColumns: 1,
    minSpace: new Size(8, 8),
    minItemSize: new Size(SIDEBAR_MEDIA_SIZE, SIDEBAR_MEDIA_SIZE),
    maxItemSize: new Size(SIDEBAR_MEDIA_SIZE, SIDEBAR_MEDIA_SIZE),
    preserveAspectRatio: true,
};

type SidebarItemsProps = {
    items: Media[];
    isFetchingNextPage: boolean;
    mediaItem: Media;
    fetchNextPage: () => void;
    isUserReviewed: (mediaItemId: string) => boolean;
    onSelectedMediaItem: (item: Media) => void;
};

export const SidebarItems = ({
    mediaItem,
    items,
    isFetchingNextPage,
    fetchNextPage,
    isUserReviewed,
    onSelectedMediaItem,
}: SidebarItemsProps) => {
    const ref = useRef(null);
    const unwrapRef = useUnwrapDOMRef(ref);

    const selectedIndex = items.findIndex((item) => item.id === mediaItem.id);

    useKeyboardNavigation({
        ref: unwrapRef,
        items,
        selectedIndex,
        onSelectedMediaItem,
    });

    return (
        <Toolbar.Container height={'100%'}>
            <Toolbar.Section height={'100%'}>
                <Flex direction={'column'} height={'100%'} gap='size-100'>
                    <SidebarMediaFilter hasMediaItems={!isEmpty(items)} />
                    <View ref={ref} width={'100%'} flex={1}>
                        <VirtualizerGridLayout
                            items={items}
                            ariaLabel='sidebar-items'
                            selectionMode='single'
                            selectedKeys={new Set([String(mediaItem.id)])}
                            layoutOptions={layoutOptions}
                            isLoadingMore={isFetchingNextPage}
                            scrollToIndex={selectedIndex}
                            onLoadMore={fetchNextPage}
                            contentItem={(item) => (
                                <SidebarMediaItem
                                    item={item}
                                    isUserReviewed={isUserReviewed(item.id)}
                                    onSelectedMediaItem={onSelectedMediaItem}
                                />
                            )}
                        />
                    </View>
                </Flex>
            </Toolbar.Section>
        </Toolbar.Container>
    );
};

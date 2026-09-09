// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentProps, ComponentType, ReactNode, useRef } from 'react';

import { AriaComponentsListBox, AriaListBoxItem, GridLayout, Loading, View, Virtualizer } from '@geti-ui/ui';
import { useLoadMore } from '@react-aria/utils';
import { GridLayoutOptions } from 'react-aria-components';

import { IsScrollingProvider } from '../../hooks/use-is-scrolling.hook';
import { useGetTargetPosition } from './use-get-target-position.hook';

import classes from './virtualizer-grid-layout.module.scss';

type AriaComponentsListBoxProps = ComponentProps<typeof AriaComponentsListBox>;

type SelectionStateOptions = {
    allowDuplicateSelectionEvents?: boolean;
    selectOnFocus?: boolean;
};

// TODO: Extends types on @geti-ui/ui
const ListBox = AriaComponentsListBox as ComponentType<AriaComponentsListBoxProps & SelectionStateOptions>;

interface GridItem {
    id: string;
    [key: string]: unknown;
}

interface VirtualizerGridLayoutProps<T extends GridItem>
    extends
        Pick<AriaComponentsListBoxProps, 'selectedKeys' | 'onSelectionChange' | 'selectionBehavior'>,
        SelectionStateOptions {
    items: T[];
    ariaLabel: string;
    scrollToIndex?: number;
    selectionMode: 'single' | 'multiple' | 'none';
    layoutOptions: GridLayoutOptions;
    isPending?: boolean;
    isLoadingMore: boolean;
    onLoadMore: () => void;
    contentItem: (item: T) => ReactNode;
    getItemId?: (item: T) => string | number;
}

const MIN_SPACE = 18; // default value for GridLayoutOptions.minSpace.height

export const VirtualizerGridLayout = <T extends GridItem>({
    items,
    ariaLabel,
    selectedKeys,
    isPending = false,
    isLoadingMore,
    selectionMode,
    selectionBehavior,
    layoutOptions,
    scrollToIndex,
    onLoadMore,
    contentItem,
    onSelectionChange,
    allowDuplicateSelectionEvents,
    selectOnFocus,
    getItemId = (item) => item.id,
}: VirtualizerGridLayoutProps<T>) => {
    const ref = useRef<HTMLDivElement | null>(null);

    // Treat `isPending` as "loading" for the purposes of auto-pagination, so we
    // don't kick off a next-page fetch while the initial load is still in
    // flight. Without this guard the gallery shows the full overlay AND the
    // inline tile loader at the same time on first render.
    useLoadMore({ isLoading: isLoadingMore || isPending, onLoadMore }, ref);

    useGetTargetPosition({
        ref,
        delay: 40,
        gap: layoutOptions.minSpace?.height ?? MIN_SPACE,
        scrollToIndex,
        callback: (top) => {
            ref.current?.scrollTo({ top, behavior: 'smooth' });
        },
    });

    return (
        <View UNSAFE_className={classes.mainContainer}>
            <IsScrollingProvider scrollRef={ref}>
                <Virtualizer layout={GridLayout} layoutOptions={layoutOptions}>
                    <ListBox
                        ref={ref}
                        layout='grid'
                        aria-label={ariaLabel}
                        className={classes.container}
                        selectedKeys={selectedKeys}
                        selectionMode={selectionMode}
                        selectionBehavior={selectionBehavior}
                        onSelectionChange={onSelectionChange}
                        allowDuplicateSelectionEvents={allowDuplicateSelectionEvents}
                        selectOnFocus={selectOnFocus}
                    >
                        {items.map((item) => {
                            const itemId = getItemId(item);

                            return (
                                <AriaListBoxItem
                                    id={itemId}
                                    key={`${ariaLabel}-${itemId}`}
                                    textValue={String(itemId)}
                                    className={classes.mediaItem}
                                >
                                    {contentItem(item)}
                                </AriaListBoxItem>
                            );
                        })}
                        {isLoadingMore && !isPending && (
                            <AriaListBoxItem id={'loader'} textValue={'loading'}>
                                <Loading mode='overlay' />
                            </AriaListBoxItem>
                        )}
                    </ListBox>
                </Virtualizer>
            </IsScrollingProvider>
            {isPending && <Loading mode='overlay' />}
        </View>
    );
};

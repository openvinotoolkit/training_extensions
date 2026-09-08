// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, ReactNode } from 'react';

import { ActionButton, Divider, Flex, Heading, Item, Menu, MenuTrigger, View } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { ENTIRE_DATASET_VIEW_ID } from 'hooks/use-dataset-view-id.hook';
import { isEmpty } from 'lodash-es';

import { DatasetView } from '../type';
import { ENTIRE_DATASET_NAME } from '../util';

import classes from './dataset-view-items-list.module.scss';

type DatasetViewItemContainerProps = {
    name: string;
    isSelected: boolean;
    onSelect: () => void;
    children: ReactNode;
};

const DatasetViewItemContainer = ({ name, isSelected, onSelect, children }: DatasetViewItemContainerProps) => {
    return (
        <li aria-label={name} aria-current={isSelected ? true : undefined} onClick={onSelect}>
            <View
                padding={'size-200'}
                borderRadius={'regular'}
                UNSAFE_className={clsx(classes.datasetViewInListItem, {
                    [classes.datasetViewListItemSelected]: isSelected,
                })}
            >
                {children}
            </View>
        </li>
    );
};

type DatasetViewItemProps = {
    datasetView: DatasetView;
    isSelected: boolean;
    onSelectDatasetView: (datasetViewId: string) => void;
    onOpenDeleteConfirmationDialog: (datasetView: DatasetView) => void;
    onOpenRenameDialog: (datasetView: DatasetView) => void;
};

const DATASET_VIEW_ITEM_OPTIONS = {
    DELETE: 'Delete',
    RENAME: 'Rename',
};

const DatasetViewItem = ({
    datasetView,
    isSelected,
    onOpenRenameDialog,
    onSelectDatasetView,
    onOpenDeleteConfirmationDialog,
}: DatasetViewItemProps) => {
    const handleAction = (key: Key) => {
        if (key === DATASET_VIEW_ITEM_OPTIONS.DELETE) {
            onOpenDeleteConfirmationDialog(datasetView);
        } else if (key === DATASET_VIEW_ITEM_OPTIONS.RENAME) {
            onOpenRenameDialog(datasetView);
        }
    };

    return (
        <DatasetViewItemContainer
            name={datasetView.name}
            isSelected={isSelected}
            onSelect={() => onSelectDatasetView(datasetView.id)}
        >
            <Flex minWidth={0} alignItems={'center'} justifyContent={'space-between'}>
                <Heading UNSAFE_className={classes.datasetViewInList}>{datasetView.name}</Heading>
                <MenuTrigger>
                    <ActionButton isQuiet aria-label={`Dataset view actions for ${datasetView.name}`}>
                        <MoreMenu />
                    </ActionButton>
                    <Menu onAction={handleAction} aria-label={'Dataset view actions menu'}>
                        <Item key={DATASET_VIEW_ITEM_OPTIONS.RENAME}>{DATASET_VIEW_ITEM_OPTIONS.RENAME}</Item>
                        <Item key={DATASET_VIEW_ITEM_OPTIONS.DELETE}>{DATASET_VIEW_ITEM_OPTIONS.DELETE}</Item>
                    </Menu>
                </MenuTrigger>
            </Flex>
        </DatasetViewItemContainer>
    );
};

type EntireDatasetViewItemProps = {
    isSelected: boolean;
    onSelectDatasetView: (datasetViewId: string | null) => void;
};

const EntireDatasetViewItem = ({ isSelected, onSelectDatasetView }: EntireDatasetViewItemProps) => {
    return (
        <DatasetViewItemContainer
            name={ENTIRE_DATASET_NAME}
            isSelected={isSelected}
            onSelect={() => onSelectDatasetView(ENTIRE_DATASET_VIEW_ID)}
        >
            <Heading UNSAFE_className={classes.datasetViewInList}>{ENTIRE_DATASET_NAME}</Heading>
        </DatasetViewItemContainer>
    );
};

type DatasetViewItemsListProps = {
    selectedDatasetViewId: string | null;
    onOpenDeleteConfirmationDialog: (datasetView: DatasetView) => void;
    datasetViews: DatasetView[];
    onOpenRenameDialog: (datasetView: DatasetView) => void;
    onSelectDatasetView: (datasetViewId: string | null) => void;
};

export const DatasetViewItemsList = ({
    selectedDatasetViewId,
    datasetViews,
    onSelectDatasetView,
    onOpenRenameDialog,
    onOpenDeleteConfirmationDialog,
}: DatasetViewItemsListProps) => {
    return (
        <ul className={classes.datasetViewsList} aria-label={'Dataset views list'}>
            <EntireDatasetViewItem
                isSelected={selectedDatasetViewId === ENTIRE_DATASET_VIEW_ID}
                onSelectDatasetView={onSelectDatasetView}
            />

            {!isEmpty(datasetViews) && <Divider size={'S'} />}

            {datasetViews.map((datasetView) => (
                <DatasetViewItem
                    key={datasetView.id}
                    datasetView={datasetView}
                    isSelected={selectedDatasetViewId === datasetView.id}
                    onSelectDatasetView={onSelectDatasetView}
                    onOpenRenameDialog={onOpenRenameDialog}
                    onOpenDeleteConfirmationDialog={onOpenDeleteConfirmationDialog}
                />
            ))}
        </ul>
    );
};

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { toast } from '@/components/toast/toast.component';
import { Content, Dialog, DialogContainer, DialogTrigger, Flex, PressableElement, Text, View } from '@geti-ui/ui';
import { ChevronDownSmall } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { ENTIRE_DATASET_VIEW_ID, useDatasetViewId } from 'hooks/use-dataset-view-id.hook';
import { isEmpty } from 'lodash-es';

import { DatasetViewItemsList } from './dataset-view-items-list/dataset-view-items-list.component';
import { DeleteDatasetViewDialog } from './delete-dataset-view.component';
import { RenameDatasetView } from './rename-dataset-view.component';
import { DatasetView } from './type';
import { ENTIRE_DATASET_NAME } from './util';

import classes from './dataset-view-selector.module.scss';

type DatasetViewsTriggerProps = {
    selectedDatasetViewName: string;
    isDisabled: boolean;
};

const DatasetViewsTrigger = ({ selectedDatasetViewName, isDisabled }: DatasetViewsTriggerProps) => {
    return (
        <PressableElement isDisabled={isDisabled}>
            <div
                role={'button'}
                aria-label={'Select dataset view'}
                aria-disabled={isDisabled}
                tabIndex={isDisabled ? -1 : 0}
            >
                <View
                    paddingX={'size-150'}
                    paddingY={'size-100'}
                    borderRadius={'regular'}
                    maxWidth={'size-2400'}
                    UNSAFE_className={clsx(classes.datasetViewsTrigger, {
                        [classes.datasetViewsTriggerDisabled]: isDisabled,
                    })}
                >
                    <Flex alignItems={'center'} gap={'size-200'}>
                        <Text UNSAFE_className={classes.datasetViewName}>{selectedDatasetViewName}</Text>

                        <ChevronDownSmall />
                    </Flex>
                </View>
            </div>
        </PressableElement>
    );
};

type DatasetViewSelectorProps = {
    datasetViews: DatasetView[];
    resetSelectedMediaIds: () => void;
};

export const DatasetViewSelector = ({ datasetViews, resetSelectedMediaIds }: DatasetViewSelectorProps) => {
    const [isDatasetViewSelectorOpen, setIsDatasetViewSelectorOpen] = useState<boolean>(false);

    const [datasetViewId, setDatasetViewId] = useDatasetViewId();
    const selectedDatasetViewName = datasetViews.find((view) => view.id === datasetViewId)?.name ?? ENTIRE_DATASET_NAME;

    const [datasetViewToBeDeleted, setDatasetViewToBeDeleted] = useState<DatasetView | null>(null);
    const [datasetViewToBeRenamed, setDatasetViewToBeRenamed] = useState<DatasetView | null>(null);

    const openDeleteConfirmationDialog = (datasetView: DatasetView) => {
        setIsDatasetViewSelectorOpen(false);
        setDatasetViewToBeDeleted(datasetView);
    };

    const openRenameDialog = (datasetView: DatasetView) => {
        setIsDatasetViewSelectorOpen(false);
        setDatasetViewToBeRenamed(datasetView);
    };

    const handleDelete = () => {
        if (datasetViewToBeDeleted?.id === datasetViewId) {
            setDatasetViewId(ENTIRE_DATASET_VIEW_ID);
        }
        toast({
            message: `Dataset view "${datasetViewToBeDeleted?.name}" has been deleted successfully.`,
            type: 'success',
        });
        setDatasetViewToBeDeleted(null);
    };

    const handleCancelDelete = () => {
        setDatasetViewToBeDeleted(null);
    };

    const handleCloseRenameDialog = () => {
        setDatasetViewToBeRenamed(null);
    };

    const onlyEntireDatasetView = isEmpty(datasetViews);

    // When the datasetViewId is invalid, i.e. not found in the datasetViews array, set it to the default view id.
    useEffect(() => {
        if (datasetViewId !== ENTIRE_DATASET_VIEW_ID && !datasetViews.some(({ id }) => id === datasetViewId)) {
            setDatasetViewId(ENTIRE_DATASET_VIEW_ID);
        }
    }, [datasetViewId, datasetViews, setDatasetViewId]);

    const selectDatasetView = (id: string | null) => {
        setIsDatasetViewSelectorOpen(false);

        if (id === datasetViewId) return;

        setDatasetViewId(id);
        resetSelectedMediaIds();
    };

    return (
        <Flex gap={'size-100'} alignItems={'center'}>
            <Text UNSAFE_className={classes.viewsTitle}>Views</Text>

            <DialogTrigger
                hideArrow
                type={'popover'}
                placement={'bottom left'}
                isOpen={isDatasetViewSelectorOpen}
                onOpenChange={setIsDatasetViewSelectorOpen}
            >
                <DatasetViewsTrigger
                    selectedDatasetViewName={selectedDatasetViewName}
                    isDisabled={onlyEntireDatasetView}
                />
                <Dialog>
                    <Content>
                        <DatasetViewItemsList
                            datasetViews={datasetViews}
                            selectedDatasetViewId={datasetViewId}
                            onOpenDeleteConfirmationDialog={openDeleteConfirmationDialog}
                            onSelectDatasetView={selectDatasetView}
                            onOpenRenameDialog={openRenameDialog}
                        />
                    </Content>
                </Dialog>
            </DialogTrigger>

            <DialogContainer onDismiss={handleCancelDelete}>
                {datasetViewToBeDeleted !== null && (
                    <DeleteDatasetViewDialog
                        datasetView={datasetViewToBeDeleted}
                        onSuccess={handleDelete}
                        onCancel={handleCancelDelete}
                    />
                )}
            </DialogContainer>
            <DialogContainer onDismiss={handleCloseRenameDialog}>
                {datasetViewToBeRenamed && (
                    <RenameDatasetView
                        datasetView={datasetViewToBeRenamed}
                        onClose={handleCloseRenameDialog}
                        datasetViews={datasetViews.filter((view) => view.id !== datasetViewToBeRenamed.id)}
                    />
                )}
            </DialogContainer>
        </Flex>
    );
};

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import { toast } from '@/components/toast/toast.component';
import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Form,
    Heading,
    Item,
    Picker,
    Text,
} from '@geti-ui/ui';
import { Info } from '@geti-ui/ui/icons';
import { DATASET_VIEW_ID_PARAM, ENTIRE_DATASET_VIEW_ID, useDatasetViewId } from 'hooks/use-dataset-view-id.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';
import { createSearchParams, Link, useLocation } from 'react-router-dom';

import { pluralizeItems } from '../../../../../../shared/util';
import { useAssignMediaToExistingDatasetView } from '../api/use-assign-media-to-existing-dataset-view';
import { SelectedMediaCount } from '../selected-media-count/selected-media-count.component';
import { DatasetView } from '../type';

import classes from './assign-to-existing-view.module.scss';

const useAssignMediaToExistingView = () => {
    const projectId = useProjectIdentifier();

    const assignToExistingViewMutation = useAssignMediaToExistingDatasetView();

    const assignToExistingView = ({
        selectedDatasetViewId,
        selectedMediaIds,
        onClose,
    }: {
        selectedDatasetViewId: string;
        selectedMediaIds: string[];
        onClose: (selectedDatasetViewId: string) => void;
    }) => {
        assignToExistingViewMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectId,
                        dataset_view_id: selectedDatasetViewId,
                    },
                },
                body: {
                    media_ids: selectedMediaIds,
                },
            },
            {
                onSuccess: () => {
                    onClose(selectedDatasetViewId);
                },
            }
        );
    };

    return {
        assignToExistingView,
        isPending: assignToExistingViewMutation.isPending,
    };
};

type AssignToExistingViewDialogProps = {
    datasetViews: DatasetView[];
    onClose: (selectedDatasetViewId?: string) => void;
    selectedMediaIds: string[];
};

const AssignToExistingViewDialog = ({ datasetViews, selectedMediaIds, onClose }: AssignToExistingViewDialogProps) => {
    const [selectedDatasetViewId, setSelectedDatasetViewId] = useState<string | null>(null);
    const { assignToExistingView, isPending } = useAssignMediaToExistingView();
    const isAssignDisabled = selectedDatasetViewId === null;

    const assignMedia = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (selectedDatasetViewId === null) {
            return;
        }

        assignToExistingView({ selectedDatasetViewId, selectedMediaIds, onClose });
    };

    return (
        <Dialog>
            <Heading>Assign to existing view</Heading>
            <Divider size={'S'} />
            <Content>
                <SelectedMediaCount count={selectedMediaIds.length} />
                <Form id={'assign-to-existing-view-form'} onSubmit={assignMedia} marginTop={'size-200'}>
                    <Picker
                        items={datasetViews}
                        label={'Assign to'}
                        placeholder={'Select a view'}
                        selectedKey={selectedDatasetViewId}
                        onSelectionChange={(viewId) => setSelectedDatasetViewId(viewId?.toString() ?? null)}
                    >
                        {(item) => <Item key={item.id}>{item.name}</Item>}
                    </Picker>
                </Form>
                <Flex gap={'size-50'} marginTop={'size-250'}>
                    <Info />
                    <Text UNSAFE_className={classes.note}>
                        This operation will not affect other media that were already assigned to this view.
                    </Text>
                </Flex>
            </Content>
            <ButtonGroup>
                <Button onPress={() => onClose()} variant={'secondary'}>
                    Close
                </Button>
                <Button
                    type={'submit'}
                    form={'assign-to-existing-view-form'}
                    variant={'accent'}
                    isPending={isPending}
                    isDisabled={isAssignDisabled}
                >
                    Assign
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

type AssignToExistingViewProps = {
    datasetViews: DatasetView[];
    selectedMediaIds: string[];
    resetSelectedMediaIds: () => void;
};

export const AssignToExistingView = ({
    datasetViews,
    selectedMediaIds,
    resetSelectedMediaIds,
}: AssignToExistingViewProps) => {
    const [datasetViewId] = useDatasetViewId();
    const [isAssignToExistingViewOpen, setIsAssignToExistingViewOpen] = useState<boolean>(false);
    const isAssignToExistingViewDisabled = isEmpty(datasetViews);
    const location = useLocation();

    const closeDialog = (selectedDatasetViewId?: string) => {
        if (selectedDatasetViewId != null) {
            const selectedDatasetView = datasetViews.find((view) => view.id === selectedDatasetViewId);
            const searchParams = createSearchParams(location.search);
            searchParams.set(DATASET_VIEW_ID_PARAM, selectedDatasetViewId);

            toast({
                id: 'assign-dataset-view-id',
                message: (
                    <Flex alignItems={'center'} wrap={'wrap'}>
                        <Text>
                            Media {pluralizeItems(selectedMediaIds.length)} assigned successfully.{' '}
                            <Link
                                to={{ pathname: location.pathname, search: searchParams.toString() }}
                                className={classes.link}
                            >
                                Open {selectedDatasetView?.name} view
                            </Link>
                        </Text>
                    </Flex>
                ),
                type: 'success',
            });
            resetSelectedMediaIds();
        }
        setIsAssignToExistingViewOpen(false);
    };

    if (datasetViewId !== ENTIRE_DATASET_VIEW_ID) {
        return null;
    }

    return (
        <>
            <Button
                variant={'primary'}
                onPress={() => setIsAssignToExistingViewOpen(true)}
                isDisabled={isAssignToExistingViewDisabled}
            >
                Assign to existing view
            </Button>
            <DialogContainer onDismiss={closeDialog}>
                {isAssignToExistingViewOpen && (
                    <AssignToExistingViewDialog
                        datasetViews={datasetViews}
                        onClose={closeDialog}
                        selectedMediaIds={selectedMediaIds}
                    />
                )}
            </DialogContainer>
        </>
    );
};

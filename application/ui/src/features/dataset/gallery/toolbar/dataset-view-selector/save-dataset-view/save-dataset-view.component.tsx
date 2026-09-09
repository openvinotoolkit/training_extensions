// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Form, Heading, TextField } from '@geti-ui/ui';
import { ENTIRE_DATASET_VIEW_ID, useDatasetViewId } from 'hooks/use-dataset-view-id.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';

import { useCreateDatasetViewMutation } from '../api/use-create-dataset-view';
import { SelectedMediaCount } from '../selected-media-count/selected-media-count.component';
import { DatasetView } from '../type';
import { DUPLICATE_DATASET_VIEW_NAME_ERROR } from '../util';

type SaveDatasetViewDialogProps = {
    onClose: (datasetViewId?: string) => void;
    selectedMediaIds: string[];
    datasetViews: DatasetView[];
};

const SaveDatasetViewDialog = ({ onClose, selectedMediaIds, datasetViews }: SaveDatasetViewDialogProps) => {
    const [viewName, setViewName] = useState<string>('');
    const projectId = useProjectIdentifier();
    const createDatasetViewMutation = useCreateDatasetViewMutation();
    const isDuplicatedName = datasetViews.some((view) => view.name === viewName.trim());

    const isSaveDisabled = isEmpty(viewName.trim()) || isDuplicatedName;

    const saveView = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        createDatasetViewMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectId,
                    },
                },
                body: {
                    name: viewName.trim(),
                    media_ids: selectedMediaIds,
                },
            },
            {
                onSuccess: ({ id }) => {
                    onClose(id);
                },
            }
        );
    };

    return (
        <Dialog>
            <Heading>Save view</Heading>
            <Divider size={'S'} />
            <Content>
                <SelectedMediaCount count={selectedMediaIds.length} />
                <Form id={'view-name-form'} onSubmit={saveView} marginTop={'size-200'}>
                    <TextField
                        // eslint-disable-next-line jsx-a11y/no-autofocus
                        autoFocus
                        label={'View name'}
                        value={viewName}
                        onChange={setViewName}
                        validationState={isDuplicatedName ? 'invalid' : undefined}
                        errorMessage={isDuplicatedName ? DUPLICATE_DATASET_VIEW_NAME_ERROR : undefined}
                    />
                </Form>
            </Content>
            <ButtonGroup>
                <Button variant={'secondary'} onPress={() => onClose()}>
                    Close
                </Button>
                <Button
                    variant={'accent'}
                    type={'submit'}
                    form={'view-name-form'}
                    isDisabled={isSaveDisabled}
                    isPending={createDatasetViewMutation.isPending}
                >
                    Save
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

type SaveDatasetViewProps = {
    selectedMediaIds: string[];
    datasetViews: DatasetView[];
    resetSelectedMediaIds: () => void;
};

export const SaveDatasetView = ({ selectedMediaIds, datasetViews, resetSelectedMediaIds }: SaveDatasetViewProps) => {
    const [datasetViewId, setDatasetViewId] = useDatasetViewId();

    const [isSaveViewDialogOpen, setIsSaveViewDialogOpen] = useState<boolean>(false);

    if (datasetViewId !== ENTIRE_DATASET_VIEW_ID) {
        return null;
    }

    const closeDialog = (newDatasetViewId?: string) => {
        if (newDatasetViewId) {
            setDatasetViewId(newDatasetViewId);
            resetSelectedMediaIds();
        }
        setIsSaveViewDialogOpen(false);
    };

    return (
        <>
            <Button variant={'primary'} onPress={() => setIsSaveViewDialogOpen(true)}>
                Save view
            </Button>
            <DialogContainer onDismiss={closeDialog}>
                {isSaveViewDialogOpen && (
                    <SaveDatasetViewDialog
                        onClose={closeDialog}
                        selectedMediaIds={selectedMediaIds}
                        datasetViews={datasetViews}
                    />
                )}
            </DialogContainer>
        </>
    );
};

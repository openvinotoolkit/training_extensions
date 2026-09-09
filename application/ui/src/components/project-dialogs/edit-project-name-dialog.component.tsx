// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Form, Heading, TextField } from '@geti-ui/ui';
import { usePatchProject } from 'hooks/api/project.hook';
import { isEmpty } from 'lodash-es';
import { useTranslation } from 'react-i18next';

import { PROJECT_NAME_MAX_LENGTH, validateProjectName } from '../../features/project/validator';
import { toast } from '../toast/toast.component';

type EditProjectNameDialogProps = {
    onClose: () => void;
    isOpen: boolean;
    projectId: string;
    projectName: string;
    projectNames: string[];
};

export const EditProjectNameDialog = ({
    onClose,
    isOpen,
    projectId,
    projectName,
    projectNames,
}: EditProjectNameDialogProps) => {
    const { t } = useTranslation();
    const patchProjectMutation = usePatchProject();
    const [newProjectName, setNewProjectName] = useState(projectName);

    const trimmedProjectName = newProjectName.trim();
    const isNameUnchanged = trimmedProjectName === projectName;
    const validationErrorMessage = validateProjectName(newProjectName, projectNames, t);
    const isSaveButtonDisabled =
        isEmpty(trimmedProjectName) ||
        isNameUnchanged ||
        patchProjectMutation.isPending ||
        validationErrorMessage !== undefined;

    const editProjectName = (newName: string) => {
        patchProjectMutation.mutate(
            {
                params: { path: { project_id: projectId } },
                body: { name: newName },
            },
            {
                onSuccess: () => {
                    onClose();
                    toast({ type: 'success', message: t('project.rename.success') });
                },
            }
        );
    };

    const handleEditProjectName = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        if (isSaveButtonDisabled) {
            return;
        }

        editProjectName(newProjectName);
    };

    return (
        <DialogContainer onDismiss={onClose}>
            {isOpen && (
                <Dialog>
                    <Heading>{t('project.rename.title')}</Heading>
                    <Divider />
                    <Content>
                        <Form onSubmit={handleEditProjectName}>
                            <TextField
                                //eslint-disable-next-line jsx-a11y/no-autofocus
                                autoFocus
                                maxLength={PROJECT_NAME_MAX_LENGTH}
                                value={newProjectName}
                                onChange={setNewProjectName}
                                width='100%'
                                aria-label={t('project.rename.fieldLabel')}
                                isReadOnly={patchProjectMutation.isPending}
                                errorMessage={validationErrorMessage}
                                validationState={validationErrorMessage === undefined ? undefined : 'invalid'}
                            />
                            <ButtonGroup align={'end'} marginTop={'size-350'}>
                                <Button
                                    variant='secondary'
                                    onPress={onClose}
                                    isDisabled={patchProjectMutation.isPending}
                                >
                                    {t('common.actions.cancel')}
                                </Button>
                                <Button
                                    type='submit'
                                    variant='accent'
                                    isDisabled={isSaveButtonDisabled}
                                    isPending={patchProjectMutation.isPending}
                                >
                                    {t('common.actions.save')}
                                </Button>
                            </ButtonGroup>
                        </Form>
                    </Content>
                </Dialog>
            )}
        </DialogContainer>
    );
};

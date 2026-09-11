// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { Model } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, AlertDialog, DialogContainer, Item, Key, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { useDeleteModel } from '../../../hooks/api/use-delete-model.hook';
import { useRenameModel } from '../../../hooks/api/use-rename-model.hook';
import { TrainingLogsDialog } from '../../../training-logs/training-logs-dialog.component';
import { hasDeletedWeights, isFailedModel, isTrainingModel } from '../../utils/utils';
import { RenameModelDialog } from '../model-row/rename-model-dialog.component';

const MODEL_ACTIONS = {
    RENAME: 'rename',
    DELETE_MODEL: 'delete_model',
    DELETE_WEIGHTS: 'delete_weights',
    VIEW_LOGS: 'view_logs',
};

enum DIALOG_TYPES {
    RENAME = 'rename',
    DELETE_MODEL = 'delete_model',
    DELETE_WEIGHTS = 'delete_weights',
    LOGS = 'logs',
}
type ModelActionsProps = {
    model: Model;
};

export const ModelActions = ({ model }: ModelActionsProps) => {
    const { t } = useTranslation();
    const projectId = useProjectIdentifier();
    const deleteModelMutation = useDeleteModel();
    const renameModelMutation = useRenameModel();

    const [isDialogOpen, setIsDialogOpen] = useState<DIALOG_TYPES | null>(null);

    const disableRename = isFailedModel(model) || isTrainingModel(model);
    const disabledKeys = [];

    if (disableRename) disabledKeys.push(MODEL_ACTIONS.RENAME);
    if (isTrainingModel(model)) disabledKeys.push(MODEL_ACTIONS.VIEW_LOGS);
    if (hasDeletedWeights(model)) disabledKeys.push(MODEL_ACTIONS.DELETE_WEIGHTS, MODEL_ACTIONS.VIEW_LOGS);

    const handleAction = (key: Key) => {
        if (key === MODEL_ACTIONS.DELETE_MODEL) {
            setIsDialogOpen(DIALOG_TYPES.DELETE_MODEL);
        } else if (key === MODEL_ACTIONS.DELETE_WEIGHTS) {
            setIsDialogOpen(DIALOG_TYPES.DELETE_WEIGHTS);
        } else if (key === MODEL_ACTIONS.RENAME) {
            setIsDialogOpen(DIALOG_TYPES.RENAME);
        } else if (key === MODEL_ACTIONS.VIEW_LOGS) {
            setIsDialogOpen(DIALOG_TYPES.LOGS);
        }
    };

    const handleRename = (newName: string) => {
        renameModelMutation.mutate(
            {
                params: { path: { project_id: projectId, model_id: model.id } },
                body: { name: newName },
            },
            {
                onSuccess: () => {
                    setIsDialogOpen(null);
                },
            }
        );
    };

    const handleDeleteModel = (filesOnly: boolean) => {
        deleteModelMutation.mutate(
            {
                params: {
                    path: { project_id: projectId, model_id: model.id },
                    query: { files_only: filesOnly },
                },
            },
            {
                onSuccess: () => setIsDialogOpen(null),
            }
        );
    };

    const modelName = model.name;

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Model actions'}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction} aria-label={'Model actions menu'} disabledKeys={disabledKeys}>
                    <Item key={MODEL_ACTIONS.RENAME}>{t('models.actions.rename')}</Item>
                    <Item key={MODEL_ACTIONS.DELETE_WEIGHTS}>{t('models.actions.deleteWeights')}</Item>
                    <Item key={MODEL_ACTIONS.DELETE_MODEL}>{t('models.actions.deleteModel')}</Item>
                    <Item key={MODEL_ACTIONS.VIEW_LOGS}>{t('models.actions.viewLogs')}</Item>
                </Menu>
            </MenuTrigger>

            <DialogContainer onDismiss={() => setIsDialogOpen(null)}>
                {isDialogOpen === DIALOG_TYPES.RENAME && (
                    <RenameModelDialog
                        currentName={model.name}
                        onRename={handleRename}
                        isPending={renameModelMutation.isPending}
                        onClose={() => setIsDialogOpen(null)}
                    />
                )}
            </DialogContainer>
            <DialogContainer onDismiss={() => setIsDialogOpen(null)}>
                {isDialogOpen === DIALOG_TYPES.DELETE_WEIGHTS && (
                    <AlertDialog
                        title={t('models.actions.deleteWeights')}
                        variant='destructive'
                        primaryActionLabel={t('models.actions.deleteWeights')}
                        onPrimaryAction={() => handleDeleteModel(true)}
                        isPrimaryActionDisabled={deleteModelMutation.isPending}
                        cancelLabel={t('models.actions.cancel')}
                    >
                        {t('models.actions.deleteWeightsDescription', { modelName })}
                    </AlertDialog>
                )}
            </DialogContainer>
            <DialogContainer onDismiss={() => setIsDialogOpen(null)}>
                {isDialogOpen === DIALOG_TYPES.DELETE_MODEL && (
                    <AlertDialog
                        title={t('models.actions.deleteModel')}
                        variant='destructive'
                        primaryActionLabel={t('models.actions.deleteModel')}
                        onPrimaryAction={() => handleDeleteModel(false)}
                        isPrimaryActionDisabled={deleteModelMutation.isPending}
                        cancelLabel={t('models.actions.cancel')}
                    >
                        {t('models.actions.deleteModelDescription', { modelName })}
                    </AlertDialog>
                )}
            </DialogContainer>
            <DialogContainer type={'fullscreen'} onDismiss={() => setIsDialogOpen(null)}>
                {isDialogOpen === DIALOG_TYPES.LOGS && <TrainingLogsDialog modelId={model.id} />}
            </DialogContainer>
        </>
    );
};

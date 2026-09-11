// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { Label } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, AlertDialog, DialogContainer, DialogTrigger, Text, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Add, Edit } from '@geti-ui/ui/icons';
import { useOverlayTriggerState } from '@react-stately/overlays';

import { useLabels } from '../use-labels.hook';
import { LabelsEditor } from './labels-editor.component';

const POPOVER_OFFSET_ALIGNMENT = 8;

type LabelsEditorPopoverProps = {
    isClassification?: boolean;
    isMultiLabel?: boolean;
    hasLabels: boolean;
};

export const LabelsEditorPopover = ({
    isClassification = false,
    isMultiLabel = false,
    hasLabels,
}: LabelsEditorPopoverProps) => {
    const { t } = useTranslation();
    const { deleteLabel } = useLabels({ isClassification, isMultiLabel });

    const popoverState = useOverlayTriggerState({});
    const deleteDialogState = useOverlayTriggerState({});
    const [labelToDelete, setLabelToDelete] = useState<Label | null>(null);

    const handleRequestDeleteLabel = (label: Label) => {
        setLabelToDelete(label);
        popoverState.close();
        deleteDialogState.open();
    };

    const handleConfirmDeleteLabel = () => {
        if (labelToDelete) {
            deleteDialogState.close();
            deleteLabel(labelToDelete.id);
            setLabelToDelete(null);
        }
    };

    const handleCancelDeleteLabel = () => {
        deleteDialogState.close();
        setLabelToDelete(null);
        popoverState.open();
    };

    const triggerAriaLabel = hasLabels ? 'Edit labels' : 'Create label';
    const triggerLabel = hasLabels ? t('labels.editor.editTrigger') : t('labels.editor.createTrigger');

    return (
        <>
            <DialogTrigger
                type='popover'
                hideArrow
                isOpen={popoverState.isOpen}
                onOpenChange={popoverState.setOpen}
                placement='bottom end'
                offset={POPOVER_OFFSET_ALIGNMENT}
                crossOffset={POPOVER_OFFSET_ALIGNMENT}
            >
                <TooltipTrigger>
                    <ActionButton isQuiet aria-label={triggerAriaLabel}>
                        {hasLabels ? (
                            <Edit />
                        ) : (
                            <>
                                <Add />
                                <Text>{triggerLabel}</Text>
                            </>
                        )}
                    </ActionButton>
                    <Tooltip>{triggerLabel}</Tooltip>
                </TooltipTrigger>

                <LabelsEditor
                    isClassification={isClassification}
                    isMultiLabel={isMultiLabel}
                    onRequestDeleteLabel={handleRequestDeleteLabel}
                    autoCreateNewLabel={!hasLabels}
                />
            </DialogTrigger>

            <DialogContainer onDismiss={handleCancelDeleteLabel}>
                {deleteDialogState.isOpen && labelToDelete && (
                    <AlertDialog
                        title={t('labels.editor.delete.title')}
                        variant={'destructive'}
                        primaryActionLabel={t('labels.editor.delete.confirm')}
                        cancelLabel={t('labels.editor.delete.cancel')}
                        onPrimaryAction={handleConfirmDeleteLabel}
                        onCancel={handleCancelDeleteLabel}
                    >
                        {t('labels.editor.delete.message', { labelName: labelToDelete.name })}
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { ActionButton, DialogContainer, Tooltip, TooltipTrigger } from '@geti-ui/ui';
import { Delete } from '@geti-ui/ui/icons';
import { isEmpty } from 'lodash-es';
import { useHotkeys } from 'react-hotkeys-hook';

import { HOTKEYS } from '../../../../shared/hotkeys-definition';
import { useDeleteMediaItem } from '../../api/use-delete-media-item';
import { AlertDialogContent } from './alert-dialog-content.component';

import classes from './delete-media-item.module.scss';

type DeleteMediaItemProps = {
    itemsIds: string[];
    onDeleted?: (deletedIds: string[]) => void;
    // Opt-in, so the hotkey does not clash with "delete annotation" in the annotator
    isHotkeyEnabled?: boolean;
};

export const DeleteMediaItem = ({ itemsIds = [], onDeleted, isHotkeyEnabled = false }: DeleteMediaItemProps) => {
    const { t } = useTranslation();
    const { deleteMedia, openDeleteDialog, closeDeleteDialog, isPending, isDeleteDialogOpen } = useDeleteMediaItem();

    const handleDelete = async () => {
        await deleteMedia(itemsIds, onDeleted);
    };

    useHotkeys(
        HOTKEYS.delete,
        openDeleteDialog,
        {
            enabled: isHotkeyEnabled && !isPending && !isEmpty(itemsIds),
            // Selected gallery items have role="option", which the library blocks by default
            enableOnFormTags: ['option'],
            preventDefault: true,
        },
        [isHotkeyEnabled, openDeleteDialog, isPending, itemsIds]
    );

    return (
        <>
            <TooltipTrigger>
                <ActionButton
                    isQuiet
                    aria-label={t('dataset.delete.mediaItem')}
                    isDisabled={isPending}
                    UNSAFE_className={classes.deleteButton}
                    onPress={openDeleteDialog}
                >
                    <Delete />
                </ActionButton>
                <Tooltip>{t('dataset.delete.mediaItem')}</Tooltip>
            </TooltipTrigger>

            <DialogContainer onDismiss={closeDeleteDialog}>
                {isDeleteDialogOpen && <AlertDialogContent itemsIds={itemsIds} onPrimaryAction={handleDelete} />}
            </DialogContainer>
        </>
    );
};

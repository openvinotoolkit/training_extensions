// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { ActionButton, DialogContainer, Item, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import { downloadFile } from '../../../../platform/download-file';
import { useDeleteMediaItem } from '../../api/use-delete-media-item';
import { AlertDialogContent } from '../delete-media-item/alert-dialog-content.component';

const MEDIA_ACTIONS = {
    DOWNLOAD: 'download',
    DELETE: 'delete',
    ANNOTATE: 'annotate',
};

type MediaItemActionsProps = {
    id: string;
    mediaUrl: string;
    mediaFileName: string;
    onDeleted?: (deletedIds: string[]) => void;
    onAnnotate: () => void;
};

export const MediaItemActions = ({ id, onDeleted, mediaUrl, mediaFileName, onAnnotate }: MediaItemActionsProps) => {
    const { closeDeleteDialog, openDeleteDialog, isDeleteDialogOpen, deleteMedia, isPending } = useDeleteMediaItem();

    const handleAction = (key: Key) => {
        if (key === MEDIA_ACTIONS.DOWNLOAD) {
            downloadFile(mediaUrl, mediaFileName, `${mediaFileName} download has started`);
        } else if (key === MEDIA_ACTIONS.DELETE) {
            openDeleteDialog();
        } else if (key === MEDIA_ACTIONS.ANNOTATE) {
            onAnnotate();
        }
    };

    const handleDeleteMedia = async () => {
        await deleteMedia([id], onDeleted);
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label={'Media actions'} isDisabled={isPending}>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction} aria-label={'Media actions menu'}>
                    <Item key={MEDIA_ACTIONS.ANNOTATE}>Annotate</Item>
                    <Item key={MEDIA_ACTIONS.DOWNLOAD}>Download</Item>
                    <Item key={MEDIA_ACTIONS.DELETE}>Delete</Item>
                </Menu>
            </MenuTrigger>
            <DialogContainer onDismiss={closeDeleteDialog}>
                {isDeleteDialogOpen && <AlertDialogContent itemsIds={[id]} onPrimaryAction={handleDeleteMedia} />}
            </DialogContainer>
        </>
    );
};

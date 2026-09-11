// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, useContext, useEffect, useMemo, useReducer } from 'react';

import { removeToast, toast } from '@/components/toast/toast.component';
import { useTranslation, type TranslateFn } from '@/i18n';
import { Button, Flex, Loading } from '@geti-ui/ui';

import { UploadDetailsDialog } from '../gallery/upload-details-dialog/upload-details-dialog.component';
import { Action, computeSummary, INITIAL_STATE, MediaUploadState, reducer } from './media-upload-reducer';

const UPLOAD_TOAST_ID = 'upload-progress-notification';
const UPLOAD_TOAST_FONT_SIZE = 'var(--spectrum-global-dimension-font-size-75)';

type MediaUploadContextValue = {
    state: MediaUploadState;
    dispatch: Dispatch<Action>;
};

const MediaUploadContext = createContext<MediaUploadContextValue | null>(null);

const buildProgressDetail = (succeeded: number, failed: number, t: TranslateFn): string => {
    const parts = [
        succeeded > 0 ? t('dataset.upload.inProgressSucceededPart', { count: succeeded }) : null,
        failed > 0 ? t('dataset.upload.inProgressFailedPart', { count: failed }) : null,
    ].filter(Boolean);

    return parts.length === 0 ? '' : `(${parts.join(', ')})`;
};

const ShowDetailsButton = ({ onPress }: { onPress: () => void }) => {
    const { t } = useTranslation();

    return (
        <Button variant={'secondary'} style={'fill'} onPress={onPress}>
            {t('dataset.upload.showDetails')}
        </Button>
    );
};

const InProgressMessage = ({
    total,
    succeeded,
    failed,
}: {
    total: number;
    succeeded: number;
    failed: number;
}): ReactNode => {
    const { t } = useTranslation();
    const detail = buildProgressDetail(succeeded, failed, t);

    return (
        <Flex alignItems={'center'} gap={'size-100'} UNSAFE_style={{ fontSize: UPLOAD_TOAST_FONT_SIZE }}>
            <Loading mode={'inline'} size={'S'} />
            <span>{`${t('dataset.upload.inProgressToast', { count: total })} ${detail}`.trim()}</span>
        </Flex>
    );
};

const showInProgressToast = (total: number, succeeded: number, failed: number, openDialog: () => void): void => {
    toast({
        id: UPLOAD_TOAST_ID,
        type: 'neutral',
        message: <InProgressMessage total={total} succeeded={succeeded} failed={failed} />,
        actionButtons: [<ShowDetailsButton key={'show-details'} onPress={openDialog} />],
        hasCloseButton: true,
        duration: Infinity,
    });
};

const showFinalToast = (succeeded: number, failed: number, openDialog: () => void, t: TranslateFn): void => {
    let text: string;

    if (failed === 0) {
        text = t('dataset.upload.uploadedSummary', { count: succeeded });
    } else if (succeeded === 0) {
        text = t('dataset.upload.failedSummary', { count: failed });
    } else {
        text = t('dataset.upload.mixedSummary', { count: succeeded, uploaded: succeeded, failed });
    }

    toast({
        id: UPLOAD_TOAST_ID,
        type: 'neutral',
        message: <span style={{ fontSize: UPLOAD_TOAST_FONT_SIZE }}>{text}</span>,
        actionButtons: [<ShowDetailsButton key={'show-details'} onPress={openDialog} />],
        hasCloseButton: true,
        duration: 5000,
    });
};

export const MediaUploadProvider = ({ children }: { children: ReactNode }) => {
    const { t } = useTranslation();
    const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

    useEffect(() => {
        return () => removeToast(UPLOAD_TOAST_ID);
    }, []);

    useEffect(() => {
        if (state.items.length === 0) return;

        const openDialog = () => dispatch({ type: 'OPEN_DIALOG' });
        const summary = computeSummary(state.items, state.isUploading);

        if (state.isUploading) {
            showInProgressToast(summary.total, summary.succeeded, summary.failed, openDialog);
        } else {
            showFinalToast(summary.succeeded, summary.failed, openDialog, t);
        }
    }, [state.items, state.isUploading, t]);

    const value = useMemo<MediaUploadContextValue>(() => ({ state, dispatch }), [state]);

    return (
        <MediaUploadContext.Provider value={value}>
            {children}
            <UploadDetailsDialog />
        </MediaUploadContext.Provider>
    );
};

export const useMediaUploadContext = (): MediaUploadContextValue => {
    const context = useContext(MediaUploadContext);

    if (context === null) {
        throw new Error('useMediaUploadContext was used outside of MediaUploadProvider');
    }

    return context;
};

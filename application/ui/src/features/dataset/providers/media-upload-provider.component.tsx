// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, useContext, useEffect, useMemo, useReducer } from 'react';

import { removeToast, toast } from '@/components/toast/toast.component';
import { Button, Flex, Loading } from '@geti-ui/ui';

import { pluralizeItems } from '../../../shared/util';
import { UploadDetailsDialog } from '../gallery/upload-details-dialog/upload-details-dialog.component';
import { Action, computeSummary, INITIAL_STATE, MediaUploadState, reducer } from './media-upload-reducer';

const UPLOAD_TOAST_ID = 'upload-progress-notification';
const UPLOAD_TOAST_FONT_SIZE = 'var(--spectrum-global-dimension-font-size-75)';

type MediaUploadContextValue = {
    state: MediaUploadState;
    dispatch: Dispatch<Action>;
};

const MediaUploadContext = createContext<MediaUploadContextValue | null>(null);

const buildProgressDetail = (succeeded: number, failed: number): string => {
    const parts = [succeeded > 0 ? `${succeeded} succeeded` : null, failed > 0 ? `${failed} failed` : null].filter(
        Boolean
    );

    return parts.length === 0 ? '' : `(${parts.join(', ')})`;
};

const ShowDetailsButton = ({ onPress }: { onPress: () => void }) => (
    <Button variant={'secondary'} style={'fill'} onPress={onPress}>
        Show details
    </Button>
);

const InProgressMessage = ({ total, detail }: { total: number; detail: string }): ReactNode => (
    <Flex alignItems={'center'} gap={'size-100'} UNSAFE_style={{ fontSize: UPLOAD_TOAST_FONT_SIZE }}>
        <Loading mode={'inline'} size={'S'} />
        <span>{`Uploading ${total} ${pluralizeItems(total)}... ${detail}`.trim()}</span>
    </Flex>
);

const showInProgressToast = (total: number, succeeded: number, failed: number, openDialog: () => void): void => {
    toast({
        id: UPLOAD_TOAST_ID,
        type: 'neutral',
        message: <InProgressMessage total={total} detail={buildProgressDetail(succeeded, failed)} />,
        actionButtons: [<ShowDetailsButton key={'show-details'} onPress={openDialog} />],
        hasCloseButton: true,
        duration: Infinity,
    });
};

const showFinalToast = (succeeded: number, failed: number, openDialog: () => void): void => {
    let text: string;

    if (failed === 0) {
        text = `Uploaded ${succeeded} ${pluralizeItems(succeeded)}`;
    } else if (succeeded === 0) {
        text = `Failed to upload ${failed} ${pluralizeItems(failed)}`;
    } else {
        text = `Uploaded ${succeeded} ${pluralizeItems(succeeded)}, ${failed} failed`;
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
            showFinalToast(summary.succeeded, summary.failed, openDialog);
        }
    }, [state.items, state.isUploading]);

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

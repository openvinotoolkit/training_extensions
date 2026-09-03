// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { v4 as uuid } from 'uuid';

import { useMediaUploadDispatch } from '../providers/media-upload-provider.component';
import { UploadFileItem } from '../providers/media-upload-reducer';

type UploadActions = {
    startUploadProgress: (files: File[]) => string[];
    setItemUploading: (itemId: string) => void;
    setItemUploaded: (itemId: string) => void;
    setItemFailed: (itemId: string, errorMessage?: string) => void;
    finishUploadProgress: () => void;
};

// Dispatch only: subscribing to the upload state here would re-render every consumer once per
// uploaded file, which freezes the page for large batches.
export const useUploadActions = (): UploadActions => {
    const dispatch = useMediaUploadDispatch();

    return useMemo(
        () => ({
            startUploadProgress: (files: File[]): string[] => {
                const newItems: UploadFileItem[] = files.map((file) => ({
                    id: uuid(),
                    name: file.name,
                    size: file.size,
                    status: 'queued',
                }));

                dispatch({ type: 'START_UPLOAD', payload: newItems });

                return newItems.map((item) => item.id);
            },
            setItemUploading: (itemId: string): void => {
                dispatch({ type: 'SET_UPLOADING', payload: { itemId } });
            },
            setItemUploaded: (itemId: string): void => {
                dispatch({ type: 'SET_UPLOADED', payload: { itemId } });
            },
            setItemFailed: (itemId: string, errorMessage?: string): void => {
                dispatch({ type: 'SET_FAILED', payload: { itemId, errorMessage } });
            },
            finishUploadProgress: (): void => {
                dispatch({ type: 'FINISH_UPLOAD' });
            },
        }),
        [dispatch]
    );
};

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMediaUploadAbortControllers, useMediaUploadDispatch } from '../providers/media-upload-context';
import { UploadFileItem } from '../providers/media-upload-reducer';

type UploadActions = {
    startUploadProgress: (files: File[]) => string[];
    setItemUploading: (itemId: string) => void;
    setItemTransferProgress: (itemId: string, bytesSent: number) => void;
    setItemUploaded: (itemId: string) => void;
    setItemFailed: (itemId: string, errorMessage?: string) => void;
    setItemCancelled: (itemId: string) => void;
    /** Signal that aborts when the item is cancelled; `undefined` once the item has settled. */
    getItemAbortSignal: (itemId: string) => AbortSignal | undefined;
    /** Forgets the item's abort controller once its upload settled. */
    releaseItem: (itemId: string) => void;
    cancelItems: (itemIds: string[]) => void;
    finishUploadProgress: () => void;
};

// Dispatch only: subscribing to the upload state here would re-render every consumer once per
// uploaded file, which freezes the page for large batches.
export const useUploadActions = (): UploadActions => {
    const dispatch = useMediaUploadDispatch();
    const abortControllers = useMediaUploadAbortControllers();

    return {
        startUploadProgress: (files: File[]): string[] => {
            const newItems: UploadFileItem[] = files.map((file) => ({
                id: crypto.randomUUID(),
                name: file.name,
                size: file.size,
                status: 'queued',
            }));

            newItems.forEach((item) => abortControllers.set(item.id, new AbortController()));
            dispatch({ type: 'START_UPLOAD', payload: newItems });

            return newItems.map((item) => item.id);
        },
        setItemUploading: (itemId: string): void => {
            dispatch({ type: 'SET_UPLOADING', payload: { itemId } });
        },
        setItemTransferProgress: (itemId: string, bytesSent: number): void => {
            dispatch({ type: 'SET_TRANSFER_PROGRESS', payload: { itemId, bytesSent } });
        },
        setItemUploaded: (itemId: string): void => {
            dispatch({ type: 'SET_UPLOADED', payload: { itemId } });
        },
        setItemFailed: (itemId: string, errorMessage?: string): void => {
            dispatch({ type: 'SET_FAILED', payload: { itemId, errorMessage } });
        },
        setItemCancelled: (itemId: string): void => {
            dispatch({ type: 'SET_CANCELLED', payload: { itemIds: [itemId] } });
        },
        getItemAbortSignal: (itemId: string): AbortSignal | undefined => abortControllers.get(itemId)?.signal,
        releaseItem: (itemId: string): void => {
            abortControllers.delete(itemId);
        },
        cancelItems: (itemIds: string[]): void => {
            itemIds.forEach((itemId) => abortControllers.get(itemId)?.abort());
            dispatch({ type: 'SET_CANCELLED', payload: { itemIds } });
        },
        finishUploadProgress: (): void => {
            dispatch({ type: 'FINISH_UPLOAD' });
        },
    };
};

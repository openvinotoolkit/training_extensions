// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TranslateFn } from '@/i18n';

export type UploadItemStatus = 'queued' | 'uploading' | 'processing' | 'uploaded' | 'failed' | 'cancelled';

// Once the transfer is complete the server creates the media item regardless, so only items that
// are still waiting or transferring can be cancelled.
export const isCancellable = (status: UploadItemStatus): boolean => status === 'queued' || status === 'uploading';

export type UploadFileItem = {
    id: string;
    name: string;
    size: number;
    status: UploadItemStatus;
    bytesSent?: number;
    errorMessage?: string;
};

export type UploadProgressSummary = {
    total: number;
    succeeded: number;
    failed: number;
    cancelled: number;
};

export type MediaUploadState = {
    items: UploadFileItem[];
    isUploading: boolean;
    isDetailsDialogOpen: boolean;
};

export const INITIAL_STATE: MediaUploadState = {
    items: [],
    isUploading: false,
    isDetailsDialogOpen: false,
};

export type Action =
    | { type: 'START_UPLOAD'; payload: UploadFileItem[] }
    | { type: 'SET_UPLOADING'; payload: { itemId: string } }
    | { type: 'SET_TRANSFER_PROGRESS'; payload: { itemId: string; bytesSent: number } }
    | { type: 'SET_UPLOADED'; payload: { itemId: string } }
    | { type: 'SET_FAILED'; payload: { itemId: string; errorMessage?: string } }
    | { type: 'SET_CANCELLED'; payload: { itemIds: string[] } }
    | { type: 'FINISH_UPLOAD' }
    | { type: 'OPEN_DIALOG' }
    | { type: 'CLOSE_DIALOG' };

export const reducer = (state: MediaUploadState, action: Action): MediaUploadState => {
    switch (action.type) {
        case 'START_UPLOAD':
            return {
                ...state,
                // Append rather than replace so the details dialog accumulates history across uploads.
                items: [...state.items, ...action.payload],
                isUploading: true,
            };
        case 'SET_UPLOADING':
            return {
                ...state,
                items: state.items.map((item) =>
                    item.id === action.payload.itemId && item.status === 'queued'
                        ? { ...item, status: 'uploading' }
                        : item
                ),
            };
        case 'SET_TRANSFER_PROGRESS':
            return {
                ...state,
                items: state.items.map((item) =>
                    item.id === action.payload.itemId && item.status === 'uploading'
                        ? {
                              ...item,
                              bytesSent: Math.min(item.size, action.payload.bytesSent),
                              status: action.payload.bytesSent >= item.size ? 'processing' : 'uploading',
                          }
                        : item
                ),
            };
        case 'SET_UPLOADED':
            return {
                ...state,
                items: state.items.map((item) =>
                    item.id === action.payload.itemId ? { ...item, status: 'uploaded', errorMessage: undefined } : item
                ),
            };
        case 'SET_FAILED':
            return {
                ...state,
                items: state.items.map((item) =>
                    item.id === action.payload.itemId
                        ? { ...item, status: 'failed', errorMessage: action.payload.errorMessage }
                        : item
                ),
            };
        case 'SET_CANCELLED': {
            const itemIds = new Set(action.payload.itemIds);

            return {
                ...state,
                items: state.items.map((item) =>
                    itemIds.has(item.id) && isCancellable(item.status) ? { ...item, status: 'cancelled' } : item
                ),
            };
        }
        case 'FINISH_UPLOAD':
            return { ...state, isUploading: false };
        case 'OPEN_DIALOG':
            return { ...state, isDetailsDialogOpen: true };
        case 'CLOSE_DIALOG':
            return { ...state, isDetailsDialogOpen: false };
        default:
            return state;
    }
};

export const computeSummary = (items: UploadFileItem[]): UploadProgressSummary => {
    const succeeded = items.filter((item) => item.status === 'uploaded').length;
    const failed = items.filter((item) => item.status === 'failed').length;
    const cancelled = items.filter((item) => item.status === 'cancelled').length;

    return {
        total: items.length,
        succeeded,
        failed,
        cancelled,
    };
};

export const formatFinalSummary = (
    t: TranslateFn,
    { succeeded, failed, cancelled }: Omit<UploadProgressSummary, 'total'>
): string => {
    if (succeeded === 0 && failed === 0 && cancelled > 0) {
        return t('dataset.upload.cancelledSummary', { count: cancelled });
    }
    if (cancelled > 0) {
        return t('dataset.upload.mixedCancelledSummary', { count: succeeded, uploaded: succeeded, failed, cancelled });
    }
    if (failed === 0) return t('dataset.upload.uploadedSummary', { count: succeeded });
    if (succeeded === 0) return t('dataset.upload.failedSummary', { count: failed });

    return t('dataset.upload.mixedSummary', { count: succeeded, uploaded: succeeded, failed });
};

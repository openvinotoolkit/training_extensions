// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type UploadItemStatus = 'queued' | 'uploading' | 'uploaded' | 'failed';

export type UploadFileItem = {
    id: string;
    name: string;
    size: number;
    status: UploadItemStatus;
    errorMessage?: string;
};

export type UploadProgressSummary = {
    total: number;
    succeeded: number;
    failed: number;
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
    | { type: 'SET_UPLOADED'; payload: { itemId: string } }
    | { type: 'SET_FAILED'; payload: { itemId: string; errorMessage?: string } }
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
                    item.id === action.payload.itemId ? { ...item, status: 'uploading' } : item
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
        case 'FINISH_UPLOAD':
            return { ...state, isUploading: false };
        case 'OPEN_DIALOG':
            return { ...state, isDetailsDialogOpen: true };
        case 'CLOSE_DIALOG':
            return { ...state, isDetailsDialogOpen: false };
    }
};

export const computeSummary = (items: UploadFileItem[]): UploadProgressSummary => {
    const succeeded = items.filter((item) => item.status === 'uploaded').length;
    const failed = items.filter((item) => item.status === 'failed').length;

    return {
        total: items.length,
        succeeded,
        failed,
    };
};

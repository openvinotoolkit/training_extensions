// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, useContext } from 'react';

import type { Action, MediaUploadState } from './media-upload-reducer';

// State and dispatch are kept apart so components that only trigger uploads (the gallery, the
// toolbar) do not re-render once per uploaded file.
export const MediaUploadStateContext = createContext<MediaUploadState | null>(null);
export const MediaUploadDispatchContext = createContext<Dispatch<Action> | null>(null);
export const IsUploadingContext = createContext<boolean | null>(null);
// One controller per queued or transferring item, so each upload can be cancelled individually.
// It is a stable mutable registry (not state), hence it never triggers re-renders.
export const MediaUploadAbortControllersContext = createContext<Map<string, AbortController> | null>(null);

export const useMediaUploadState = (): MediaUploadState => {
    const context = useContext(MediaUploadStateContext);

    if (context === null) {
        throw new Error('useMediaUploadState was used outside of MediaUploadProvider');
    }

    return context;
};

export const useMediaUploadDispatch = (): Dispatch<Action> => {
    const context = useContext(MediaUploadDispatchContext);

    if (context === null) {
        throw new Error('useMediaUploadDispatch was used outside of MediaUploadProvider');
    }

    return context;
};

export const useIsUploading = (): boolean => {
    const context = useContext(IsUploadingContext);

    if (context === null) {
        throw new Error('useIsUploading was used outside of MediaUploadProvider');
    }

    return context;
};

export const useMediaUploadAbortControllers = (): Map<string, AbortController> => {
    const context = useContext(MediaUploadAbortControllersContext);

    if (context === null) {
        throw new Error('useMediaUploadAbortControllers was used outside of MediaUploadProvider');
    }

    return context;
};

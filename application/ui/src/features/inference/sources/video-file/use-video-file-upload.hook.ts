// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import type { PrepareFormData } from '../hooks/use-source-action.hook';
import { prepareVideoFileFormData } from './utils';

export type VideoFileUploadProgress = { bytesSent: number; bytesTotal: number };

export type VideoFileUpload = {
    /** Progress of the ongoing video transfer, or `null` when no transfer is running. */
    progress: VideoFileUploadProgress | null;
    prepareFormData: PrepareFormData;
    /** Cancels the ongoing transfer; the source is then not saved. */
    cancel: () => void;
};

export const useVideoFileUpload = (): VideoFileUpload => {
    const [progress, setProgress] = useState<VideoFileUploadProgress | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Leaving the form mid-transfer must not keep uploading (and then save) the source in the background.
    useEffect(() => {
        return () => {
            abortControllerRef.current?.abort();
        };
    }, []);

    const prepareFormData: PrepareFormData = async (formData: FormData) => {
        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        try {
            return await prepareVideoFileFormData(formData, {
                signal: abortController.signal,
                onProgress: (bytesSent, bytesTotal) => setProgress({ bytesSent, bytesTotal }),
            });
        } finally {
            abortControllerRef.current = null;
            setProgress(null);
        }
    };

    return {
        progress,
        prepareFormData,
        cancel: () => abortControllerRef.current?.abort(),
    };
};

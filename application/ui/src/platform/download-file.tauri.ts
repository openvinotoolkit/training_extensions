// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';
import { i18n } from '@/i18n';
import { save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';

export const downloadFile = (url: string, name?: string, startedMessage?: string): void => {
    void saveDownload(name ?? getFallbackFilename(url), startedMessage, async () => {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`download failed: ${response.status} ${response.statusText}`);
        }

        return new Uint8Array(await response.arrayBuffer());
    });
};

export const downloadBlob = (blob: Blob, name: string, startedMessage?: string): void => {
    void saveDownload(name, startedMessage, async () => new Uint8Array(await blob.arrayBuffer()));
};

const saveDownload = async (
    filename: string,
    startedMessage: string | undefined,
    readData: () => Promise<Uint8Array>
): Promise<void> => {
    try {
        const selectedPath = await save({
            defaultPath: filename,
            filters: [getFilterForFilename(filename)],
        });

        if (selectedPath === null) {
            return;
        }

        if (startedMessage !== undefined) {
            toast({ type: 'info', message: startedMessage });
        }

        await writeFile(selectedPath, await readData());
    } catch (error: unknown) {
        console.error('[tauri downloadFile] failed', error);
        toast({ type: 'error', message: i18n.t('downloads.failure.genericError') });
    }
};

const getFilterForFilename = (filename: string): { name: string; extensions: string[] } => {
    const extension = filename.split('.').pop();

    if (extension === undefined || extension === '' || extension === filename) {
        return { name: i18n.t('downloads.saveDialog.allFiles'), extensions: ['*'] };
    }

    return {
        name: i18n.t('downloads.saveDialog.fileTypeFilter', { extension: extension.toUpperCase() }),
        extensions: [extension.toLowerCase()],
    };
};

const getFallbackFilename = (url: string): string => {
    if (url.startsWith('blob:')) {
        return 'download';
    }

    // Asset download URLs end with /binary, so the previous segment is usually the item id.
    const segments = new URL(url, window.location.origin).pathname.split('/').filter(Boolean);

    return segments.at(-2) ?? 'download';
};

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Blob as NodeBlob } from 'node:buffer';

import { downloadBlob, downloadFile } from './download-file.tauri';

// jsdom's Blob does not implement arrayBuffer(), unlike the Tauri webviews.
const createBlob = (content: string) => new NodeBlob([content]) as unknown as Blob;

const { save, writeFile } = vi.hoisted(() => ({
    save: vi.fn<(options: unknown) => Promise<string | null>>(),
    writeFile: vi.fn<(path: string, data: Uint8Array) => Promise<void>>(),
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({ save }));
vi.mock('@tauri-apps/plugin-fs', () => ({ writeFile }));
vi.mock('@/components/toast/toast.component', () => ({ toast: vi.fn() }));

describe('download-file (tauri)', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        save.mockResolvedValue('/tmp/selected');
        writeFile.mockResolvedValue();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('downloadBlob writes the blob bytes to the selected path without fetching', async () => {
        const fetchSpy = vi.spyOn(globalThis, 'fetch');

        downloadBlob(createBlob('line-1\nline-2'), 'job-logs.log');

        await vi.waitFor(() => expect(writeFile).toHaveBeenCalled());
        expect(save).toHaveBeenCalledWith(expect.objectContaining({ defaultPath: 'job-logs.log' }));
        expect(new TextDecoder().decode(writeFile.mock.calls[0][1])).toBe('line-1\nline-2');
        expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('downloadFile fetches the url and writes the response bytes', async () => {
        vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('model-bytes'));

        downloadFile('/api/models/model-1/binary', 'model.zip');

        await vi.waitFor(() => expect(writeFile).toHaveBeenCalled());
        expect(fetch).toHaveBeenCalledWith('/api/models/model-1/binary');
        expect(new TextDecoder().decode(writeFile.mock.calls[0][1])).toBe('model-bytes');
    });

    it('does not read the data when the save dialog is cancelled', async () => {
        save.mockResolvedValue(null);
        const blob = createBlob('ignored');
        const arrayBufferSpy = vi.spyOn(blob, 'arrayBuffer');

        downloadBlob(blob, 'job-logs.log');

        await vi.waitFor(() => expect(save).toHaveBeenCalled());
        expect(arrayBufferSpy).not.toHaveBeenCalled();
        expect(writeFile).not.toHaveBeenCalled();
    });
});

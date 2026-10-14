// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { openUrl } from '@tauri-apps/plugin-opener';

export const openExternalUrl = (url: string): void => {
    // Rejects when the URL is missing from the `opener:allow-open-url` allowlist in
    // src-tauri/capabilities/default.json.
    void openUrl(url).catch((error) => {
        console.error('[tauri] failed to open url', url, error);
    });
};

// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const openExternalUrl = (url: string): void => {
    window.open(url, '_blank', 'noopener,noreferrer');
};

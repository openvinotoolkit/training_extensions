// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';

export const downloadFile = (url: string, name?: string, startedMessage?: string): void => {
    const link = document.createElement('a');

    link.href = url;
    if (name !== undefined) {
        link.download = name;
    }
    link.hidden = true;
    link.click();

    if (startedMessage !== undefined) {
        toast({ type: 'info', message: startedMessage });
    }
};

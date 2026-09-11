// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@/components/toast/toast.component';
import { isEmpty } from 'lodash-es';

export const useClipboard = () => {
    const copy = (text: string, successMessage = 'Copied Successfully', errorMessage = 'Copy failed') =>
        navigator.clipboard
            .writeText(text)
            .then(() => !isEmpty(successMessage) && toast({ message: successMessage, type: 'info' }))
            .catch(() => toast({ message: errorMessage, type: 'error' }));

    return { copy };
};

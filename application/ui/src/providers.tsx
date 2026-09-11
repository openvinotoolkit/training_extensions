// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@geti-ui/ui';
import { QueryClientProvider } from '@tanstack/react-query';
import { RouterProvider } from 'react-router-dom';

import { useDocumentLanguage } from './i18n/use-document-language.hook';
import { queryClient } from './query-client/query-client';
import { router } from './router';

export const Providers = () => {
    const locale = useDocumentLanguage();

    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider router={router} locale={locale}>
                <RouterProvider
                    router={router}
                    future={{
                        v7_startTransition: true,
                    }}
                />
            </ThemeProvider>
        </QueryClientProvider>
    );
};

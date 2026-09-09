// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, type ReactNode } from 'react';

import { Toast } from '@/components/toast/toast.component';
import { IntelBrandedLoading, ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
    render as rtlRender,
    renderHook as rtlRenderHook,
    RenderOptions as RTLRenderOptions,
} from '@testing-library/react';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';

import { paths } from '../constants/paths';
import { createQueryClient } from '../query-client/query-client';

type RenderOptions = RTLRenderOptions & {
    route?: string;
    path?: string;
    queryClient?: QueryClient;
};

const TestProviders = ({ children, queryClient }: { children: ReactNode; queryClient: QueryClient }) => {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <Suspense fallback={<IntelBrandedLoading />}>{children}</Suspense>
                <Toast />
            </ThemeProvider>
        </QueryClientProvider>
    );
};

const createTestRouter = (children: ReactNode, options: RenderOptions, queryClient: QueryClient) => {
    const route = options.route ?? paths.project.details({ projectId: '123' });
    const path = options.path ?? paths.project.details.pattern;

    return createMemoryRouter(
        [
            {
                path,
                element: <TestProviders queryClient={queryClient}>{children}</TestProviders>,
            },
        ],
        {
            initialEntries: [route],
            initialIndex: 0,
        }
    );
};

export const render = (ui: ReactNode, options: RenderOptions = {}) => {
    const testQueryClient = options.queryClient ?? createQueryClient();
    const router = createTestRouter(ui, options, testQueryClient);

    return rtlRender(
        <RouterProvider
            router={router}
            future={{
                v7_startTransition: true,
            }}
        />
    );
};

export const renderHook = <TProps, TResult>(callback: (props: TProps) => TResult, options: RenderOptions = {}) => {
    const testQueryClient = options.queryClient ?? createQueryClient();

    const Wrapper = ({ children }: { children: ReactNode }) => {
        const wrappedChildren = options.wrapper ? <options.wrapper>{children}</options.wrapper> : children;
        const router = createTestRouter(wrappedChildren, options, testQueryClient);

        return (
            <RouterProvider
                router={router}
                future={{
                    v7_startTransition: true,
                }}
            />
        );
    };

    return rtlRenderHook(callback, { wrapper: Wrapper });
};

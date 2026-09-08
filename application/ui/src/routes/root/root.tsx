// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense } from 'react';

import { $api } from '@/api';
import { Toast } from '@/components/toast/toast.component';
import { Flex, Heading, Loading } from '@geti-ui/ui';
import { Outlet } from 'react-router-dom';

import { LicenseCheck } from '../../features/license/license-check.component';
import { ServerErrorFallback } from './server-error-fallback.component';

const REFETCH_INTERVAL = 5000;
const MAX_RETRIES = 30;
const retryDelay = (attempt: number) => Math.min(1000 * 2 ** attempt, 5000);

const HealthCheck = ({ children }: { children: ReactNode }) => {
    const { data, isPending, isError } = $api.useQuery('get', '/health', undefined, {
        retry: MAX_RETRIES,
        retryDelay,
        refetchInterval: (query) => {
            return query.state.data?.status === 'ok' ? false : REFETCH_INTERVAL;
        },
    });

    if (isPending) {
        return (
            <Flex direction={'column'} justifyContent={'center'} alignItems={'center'} height={'100vh'}>
                <Loading variant={'intel'} mode={'inline'} />
                <Heading bottom={'size-4600'} level={2}>
                    Loading...Please wait.
                </Heading>
            </Flex>
        );
    }

    if (isError) {
        return <ServerErrorFallback />;
    }

    if (data?.status === 'ok') {
        return children;
    }

    return <Loading variant={'intel'} />;
};

export const RootLayout = () => {
    return (
        <Suspense fallback={<Loading variant={'intel'} />}>
            <HealthCheck>
                <LicenseCheck>
                    <Outlet />
                </LicenseCheck>
            </HealthCheck>
            <div data-react-aria-top-layer='true'>
                <Toast />
            </div>
        </Suspense>
    );
};

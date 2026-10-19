// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useEffectEvent } from 'react';

import { $api } from '@/api';
import { toast } from '@/components/toast/toast.component';
import { useTranslation } from '@/i18n';
import { isFunction } from 'lodash-es';

import { isNonEmptyString } from '../../../shared/util';
import { isInvalidJob, isJobDone, isJobFailed } from '../util';
import { useStreamJobDetail } from './jobs.hook';

type UseImportJobStatusProps = {
    jobId: string | null | undefined;
    onError?: (error?: unknown) => void;
    onSuccess?: () => void;
};

export const useImportJobStatus = ({ jobId, onError, onSuccess }: UseImportJobStatusProps) => {
    const { t } = useTranslation();

    useStreamJobDetail(jobId);

    const response = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: jobId ?? '' } } },
        {
            enabled: isNonEmptyString(jobId),
        }
    );

    const handleInvalidJob = useEffectEvent((error: typeof response.error) => {
        isFunction(onError) && onError(error);
        toast({ type: 'error', message: `${t('dataset.import.genericError')} ${error?.detail}` });
    });
    const handleFailedJob = useEffectEvent((message: string | undefined | null) => {
        isFunction(onError) && onError();
        toast({ type: 'error', message: `${t('dataset.import.genericError')} ${message}` });
    });
    const handleDoneJob = useEffectEvent(() => {
        isFunction(onSuccess) && onSuccess();
    });

    useEffect(() => {
        if (response.isError && isInvalidJob(response.error)) {
            handleInvalidJob(response.error);
        }
    }, [response.error, response.isError]);

    useEffect(() => {
        if (isJobFailed(response.data)) {
            handleFailedJob(response.data?.message);
        }
    }, [response.data]);

    useEffect(() => {
        if (isJobDone(response.data)) {
            handleDoneJob();
        }
    }, [response.data]);

    return response;
};

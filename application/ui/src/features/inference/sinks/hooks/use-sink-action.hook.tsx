// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import type { SinkConfig } from '@/api/types';
import { toast } from '@/components/toast/toast.component';
import { useTranslation } from '@/i18n';
import { isFunction } from 'lodash-es';

import { useSinkMutation } from './use-sink-mutation.hook';

interface useSinkActionProps<T> {
    config: Awaited<T>;
    isNewSink: boolean;
    onSaved?: (sink_id: string) => void;
    bodyFormatter: (formData: FormData) => T;
}

export const useSinkAction = <T extends SinkConfig>({
    config,
    isNewSink,
    onSaved,
    bodyFormatter,
}: useSinkActionProps<T>) => {
    const { t } = useTranslation();
    const addOrUpdateSink = useSinkMutation(isNewSink);

    return useActionState<T, FormData>(async (_prevState: T, formData: FormData) => {
        const body = bodyFormatter(formData);

        try {
            const sink_id = await addOrUpdateSink(body);

            toast({
                type: 'success',
                message: isNewSink ? t('inference.sinks.form.createSuccess') : t('inference.sinks.form.updateSuccess'),
            });

            isFunction(onSaved) && onSaved(sink_id);

            return { ...body, id: sink_id };
        } catch (error: unknown) {
            const details = (error as { detail?: string })?.detail;

            toast({
                type: 'error',
                message: t('inference.sinks.form.saveError', {
                    details: details ?? t('inference.sinks.form.saveErrorFallback'),
                }),
            });
        }

        return body;
    }, config);
};

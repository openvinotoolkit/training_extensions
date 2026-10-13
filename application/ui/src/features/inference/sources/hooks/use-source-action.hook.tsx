// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { isAbortError } from '@/api';
import type { SourceConfigPayload } from '@/api/types';
import { toast } from '@/components/toast/toast.component';
import { useTranslation } from '@/i18n';
import { isFunction } from 'lodash-es';

import { getErrorMessage } from '../../../../query-client/query-client';
import { useSourceMutation } from './use-source-mutation.hook';

// Optionally returns a rollback that undoes its side effects (e.g. an upload) if the source is not saved.
export type PrepareFormData = (formData: FormData) => Promise<(() => Promise<void>) | undefined>;

interface useSourceActionProps<T> {
    config: Awaited<T>;
    isNewSource: boolean;
    onSaved?: (source_id: string) => void;
    bodyFormatter: (formData: FormData) => T;
    prepareFormData?: PrepareFormData;
}

export const useSourceAction = <T extends SourceConfigPayload>({
    config,
    isNewSource,
    onSaved,
    bodyFormatter,
    prepareFormData,
}: useSourceActionProps<T>) => {
    const { t } = useTranslation();
    const addOrUpdateSource = useSourceMutation(isNewSource);

    return useActionState<T, FormData>(async (prevState: T, formData: FormData) => {
        try {
            const rollback = await prepareFormData?.(formData);

            const body = bodyFormatter(formData);
            const source_id = await addOrUpdateSource(body).catch((error: unknown) => {
                // Best effort: a failed rollback must not mask the original save error.
                void rollback?.().catch(() => undefined);
                throw error;
            });

            toast({
                type: 'success',
                message: isNewSource
                    ? t('inference.sources.form.createSuccess')
                    : t('inference.sources.form.updateSuccess'),
            });

            isFunction(onSaved) && onSaved(source_id);
            return { ...body, id: source_id };
        } catch (error: unknown) {
            // The user cancelled the file transfer: keep the form as it is, without an error.
            if (isAbortError(error)) {
                return prevState;
            }

            const details = getErrorMessage(error);

            toast({
                type: 'error',
                message: t('inference.sources.form.saveError', {
                    details: details ?? t('inference.connection.saveErrorFallback'),
                }),
            });
        }

        return prevState;
    }, config);
};

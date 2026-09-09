// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import type { SourceConfigPayload } from '@/api/types';
import { toast } from '@/components/toast/toast.component';
import { isFunction } from 'lodash-es';

import { useSourceMutation } from './use-source-mutation.hook';

interface useSourceActionProps<T> {
    config: Awaited<T>;
    isNewSource: boolean;
    onSaved?: (source_id: string) => void;
    bodyFormatter: (formData: FormData) => T;
    prepareFormData?: (formData: FormData) => Promise<void>;
}

export const useSourceAction = <T extends SourceConfigPayload>({
    config,
    isNewSource,
    onSaved,
    bodyFormatter,
    prepareFormData,
}: useSourceActionProps<T>) => {
    const addOrUpdateSource = useSourceMutation(isNewSource);

    return useActionState<T, FormData>(async (prevState: T, formData: FormData) => {
        try {
            await prepareFormData?.(formData);

            const body = bodyFormatter(formData);
            const source_id = await addOrUpdateSource(body);

            toast({
                type: 'success',
                message: `Source configuration ${isNewSource ? 'created' : 'updated'} successfully.`,
            });

            isFunction(onSaved) && onSaved(source_id);
            return { ...body, id: source_id };
        } catch (error: unknown) {
            const details = (error as { detail?: string })?.detail;

            toast({
                type: 'error',
                message: `Failed to save source configuration, ${details ?? 'please try again'}`,
            });
        }

        return prevState;
    }, config);
};

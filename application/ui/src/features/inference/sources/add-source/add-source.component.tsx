// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { SourceConfigPayload } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Button, Flex, Form } from '@geti-ui/ui';

import { useConnectSourceToPipeline } from '../../../../hooks/api/pipeline.hook';
import { useSourceAction } from '../hooks/use-source-action.hook';

interface AddSourceProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
    prepareFormData?: (formData: FormData) => Promise<void>;
}

export const AddSource = <T extends SourceConfigPayload>({
    config,
    onSaved,
    bodyFormatter,
    prepareFormData,
    componentFields,
}: AddSourceProps<T>) => {
    const { t } = useTranslation();
    const connectToPipelineMutation = useConnectSourceToPipeline();

    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: true,
        onSaved: async (sourceId) => {
            await connectToPipelineMutation(sourceId);
            onSaved();
        },
        bodyFormatter,
        prepareFormData,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <Flex gap={'size-200'} direction={'column'}>
                <>{componentFields(state)}</>

                <Button type='submit' isDisabled={isPending} UNSAFE_style={{ maxWidth: 'fit-content' }}>
                    {t('inference.sources.add.submit')}
                </Button>
            </Flex>
        </Form>
    );
};

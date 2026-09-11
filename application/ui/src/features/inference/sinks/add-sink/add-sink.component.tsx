// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { SinkConfig } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Button, Form } from '@geti-ui/ui';

import { usePatchPipeline } from '../../../../hooks/api/pipeline.hook';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { useSinkAction } from '../hooks/use-sink-action.hook';

interface AddSinkProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
}

export const AddSink = <T extends SinkConfig>({ config, onSaved, bodyFormatter, componentFields }: AddSinkProps<T>) => {
    const { t } = useTranslation();
    const pipeline = usePatchPipeline();
    const project_id = useProjectIdentifier();

    const [state, submitAction, isPending] = useSinkAction({
        config,
        isNewSink: true,
        onSaved: async (sink_id) => {
            await pipeline.mutateAsync({ params: { path: { project_id } }, body: { sink_id } });
            onSaved();
        },
        bodyFormatter,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <>{componentFields(state)}</>

            <Button
                type='submit'
                isDisabled={isPending || pipeline.isPending}
                UNSAFE_style={{ maxWidth: 'fit-content' }}
            >
                {t('inference.sinks.add.submit')}
            </Button>
        </Form>
    );
};

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import type { SinkConfig } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, Button, ButtonGroup, Divider, Flex, Form, Text, View } from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';

import { useConnectSinkToPipeline } from '../../../../hooks/api/pipeline.hook';
import { testSinkQueryOptions } from '../api/use-test-sink';
import { useSinkAction } from '../hooks/use-sink-action.hook';

import classes from './edit-sink.module.scss';

interface EditSinkProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    onBackToList: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
    isConnected: boolean;
}

export const EditSink = <T extends SinkConfig>({
    config,
    onSaved,
    onBackToList,
    bodyFormatter,
    componentFields,
    isConnected,
}: EditSinkProps<T>) => {
    const { t } = useTranslation();
    const connectToPipeline = useRef(false);
    const connectToPipelineMutation = useConnectSinkToPipeline();
    const queryClient = useQueryClient();

    const [state, submitAction, isPending] = useSinkAction({
        config,
        isNewSink: false,
        onSaved: async (sinkId) => {
            connectToPipeline.current && (await connectToPipelineMutation(sinkId));
            connectToPipeline.current = false;
            onSaved();
            void queryClient.fetchQuery(testSinkQueryOptions(sinkId)).catch(() => undefined);
        },
        bodyFormatter,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <Flex gap={'size-100'} alignItems={'center'} marginTop={'0px'}>
                <ActionButton isQuiet onPress={onBackToList}>
                    <Back />
                </ActionButton>

                <Text>{t('inference.sinks.edit.title')}</Text>
            </Flex>

            <View UNSAFE_className={classes.container}>
                <>{componentFields(state)}</>
            </View>
            <Divider size='S' marginY={'size-200'} />

            <ButtonGroup marginTop={'0px'}>
                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = false)}
                >
                    {t('inference.sinks.edit.save')}
                </Button>

                {!isConnected && (
                    <Button
                        type='submit'
                        isDisabled={isPending}
                        UNSAFE_style={{ maxWidth: 'fit-content' }}
                        onPress={() => (connectToPipeline.current = true)}
                    >
                        {t('inference.sinks.edit.saveAndConnect')}
                    </Button>
                )}
            </ButtonGroup>
        </Form>
    );
};

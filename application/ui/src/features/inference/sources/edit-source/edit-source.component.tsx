// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import type { SourceConfigPayload } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, Button, ButtonGroup, Divider, Flex, Form, Text, View } from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';
import { useQueryClient } from '@tanstack/react-query';
import { useConnectSourceToPipeline } from 'hooks/api/pipeline.hook';

import { testSourceQueryOptions } from '../api/use-test-source';
import { useSourceAction } from '../hooks/use-source-action.hook';

import classes from './edit-source.module.scss';

interface EditSourceProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    onBackToList: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
    prepareFormData?: (formData: FormData) => Promise<void>;
    isConnected: boolean;
}

export const EditSource = <T extends SourceConfigPayload>({
    config,
    onSaved,
    onBackToList,
    bodyFormatter,
    prepareFormData,
    componentFields,
    isConnected,
}: EditSourceProps<T>) => {
    const { t } = useTranslation();
    const connectToPipeline = useRef(false);
    const connectToPipelineMutation = useConnectSourceToPipeline();
    const queryClient = useQueryClient();

    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: false,
        onSaved: async (sourceId) => {
            connectToPipeline.current && (await connectToPipelineMutation(sourceId));
            connectToPipeline.current = false;
            onSaved();
            void queryClient.fetchQuery(testSourceQueryOptions(sourceId)).catch(() => undefined);
        },
        bodyFormatter,
        prepareFormData,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <Flex gap={'size-100'} alignItems={'center'} marginTop={'0px'}>
                <ActionButton isQuiet onPress={onBackToList}>
                    <Back />
                </ActionButton>

                <Text>{t('inference.sources.edit.title')}</Text>
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
                    {t('inference.sources.edit.save')}
                </Button>

                {!isConnected && (
                    <Button
                        type='submit'
                        isDisabled={isPending}
                        UNSAFE_style={{ maxWidth: 'fit-content' }}
                        onPress={() => (connectToPipeline.current = true)}
                    >
                        {t('inference.sources.edit.saveAndConnect')}
                    </Button>
                )}
            </ButtonGroup>
        </Form>
    );
};

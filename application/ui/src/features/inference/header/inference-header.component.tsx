// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { useTranslation } from '@/i18n';
import { Flex, View } from '@geti-ui/ui';

import { ActiveModel } from './active-model.component';
import { TogglePipelineButton } from './toggle-pipeline-button.component';

export const Header = () => {
    const { t } = useTranslation();

    return (
        <View
            backgroundColor='gray-100'
            gridArea='toolbar'
            padding='size-200'
            UNSAFE_style={{
                fontSize: '12px',
                color: 'var(--spectrum-global-color-gray-800)',
            }}
        >
            <Flex height='100%' gap='size-200' alignItems={'center'}>
                <Suspense fallback={t('inference.pipeline.activeModel.loading')}>
                    <ActiveModel />
                </Suspense>

                <Flex marginStart='auto' gap={'size-100'}>
                    <TogglePipelineButton />
                </Flex>
            </Flex>
        </View>
    );
};

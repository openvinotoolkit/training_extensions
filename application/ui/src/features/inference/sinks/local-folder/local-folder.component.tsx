// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { LocalFolderSinkConfig } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, TextField } from '@geti-ui/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitFields } from '../rate-limit/rate-limit-fields.component';

type LocalFolderProps = {
    defaultState: LocalFolderSinkConfig;
};

export const LocalFolder = ({ defaultState }: LocalFolderProps) => {
    const { t } = useTranslation();

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />

            <Flex gap='size-200'>
                <TextField
                    label={t('inference.sinks.fields.name')}
                    name='name'
                    defaultValue={defaultState.name || t('inference.sinks.defaultNames.localFolder')}
                />
            </Flex>

            <Flex>
                <RateLimitFields rateLimit={defaultState.rate_limit} />
            </Flex>

            <Flex gap='size-50'>
                <TextField
                    isRequired
                    width={'100%'}
                    label={t('inference.sinks.fields.folderPath')}
                    name='folder_path'
                    defaultValue={defaultState.folder_path}
                />
            </Flex>

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};

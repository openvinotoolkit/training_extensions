// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ImagesFolderSourceConfig } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, Switch, TextField } from '@geti-ui/ui';

type ImageFolderProps = {
    defaultState?: ImagesFolderSourceConfig;
};

export const ImageFolder = ({ defaultState }: ImageFolderProps) => {
    const { t } = useTranslation();

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField
                width={'100%'}
                label={t('inference.sources.fields.name')}
                name='name'
                defaultValue={defaultState?.name}
            />

            <Flex direction='row' gap='size-200'>
                <TextField
                    isRequired
                    flex='1'
                    label={t('inference.sources.fields.imagesFolderPath')}
                    name='images_folder_path'
                    defaultValue={defaultState?.images_folder_path}
                />
            </Flex>

            <Switch
                aria-label='ignore existing images'
                name='ignore_existing_images'
                defaultSelected={defaultState?.ignore_existing_images}
                key={defaultState?.ignore_existing_images ? 'true' : 'false'}
            >
                {t('inference.sources.fields.ignoreExistingImages')}
            </Switch>
        </Flex>
    );
};

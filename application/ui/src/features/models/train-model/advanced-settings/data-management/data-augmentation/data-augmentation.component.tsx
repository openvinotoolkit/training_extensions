// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import type { TrainingConfiguration } from '@/api/types';
import { useTranslation } from '@/i18n';

import { Accordion } from '../../components/accordion/accordion.component';
import { DataAugmentationParametersList } from './data-augmentation-parameters-list.component';
import { DataAugmentationConfigurationParameters, isDataAugmentationEnabled } from './utils';

type DataAugmentationProps = {
    dataAugmentationParameters: DataAugmentationConfigurationParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const DataAugmentation = ({
    dataAugmentationParameters,
    onTrainingConfigurationChange,
}: DataAugmentationProps) => {
    const { t } = useTranslation();
    const isEnabled = isDataAugmentationEnabled(dataAugmentationParameters);

    return (
        <Accordion>
            <Accordion.Title>
                {t('models.training.dataManagement.augmentation.title')}
                <Accordion.Tag ariaLabel={'Data augmentation tag'}>
                    {isEnabled
                        ? t('models.training.dataManagement.augmentation.enabledYes')
                        : t('models.training.dataManagement.augmentation.enabledNo')}
                </Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{dataAugmentationParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <DataAugmentationParametersList
                    dataAugmentationParameters={dataAugmentationParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            </Accordion.Content>
        </Accordion>
    );
};

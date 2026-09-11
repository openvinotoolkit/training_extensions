// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import type { TrainingConfiguration } from '@/api/types';
import { useTranslation } from '@/i18n';
import { isEqual } from 'lodash-es';

import { Accordion } from '../../components/accordion/accordion.component';
import { LearningParametersListContainer } from './learning-parameters-list.component';
import { LearningConfigurationGroup } from './utils';

type LearningParametersProps = {
    learningParameters: LearningConfigurationGroup;
    defaultLearningParameters?: LearningConfigurationGroup;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const LearningParameters = ({
    learningParameters,
    defaultLearningParameters,
    onTrainingConfigurationChange,
}: LearningParametersProps) => {
    const { t } = useTranslation();
    const tag = isEqual(learningParameters, defaultLearningParameters)
        ? t('models.training.learning.tagDefault')
        : t('models.training.learning.tagModified');

    return (
        <Accordion>
            <Accordion.Title>
                {t('models.training.learning.title')}
                <Accordion.Tag ariaLabel={'Learning parameters tag'}>{tag}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{learningParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <LearningParametersListContainer
                    learningParameters={learningParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            </Accordion.Content>
        </Accordion>
    );
};

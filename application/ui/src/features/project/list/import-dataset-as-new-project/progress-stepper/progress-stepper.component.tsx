// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { dimensionValue, Divider, Grid, Text } from '@geti-ui/ui';

import { ImportDatasetAsNewProjectState } from '../../../../dataset/import-export/import-dataset/util';

import classes from './progress-stepper.module.scss';

type ProgressStepperProps = {
    currentStep: ImportDatasetAsNewProjectState;
};

const isLabelMapping = (step: ImportDatasetAsNewProjectState) => ['labelMapping'].includes(step);
const isTaskTypeSelection = (step: ImportDatasetAsNewProjectState) => ['taskTypeSelection'].includes(step);
const isUploadingOrPreparing = (step: ImportDatasetAsNewProjectState) => ['uploading', 'preparing'].includes(step);

export const ProgressStepper = ({ currentStep }: ProgressStepperProps) => {
    const { t } = useTranslation();

    return (
        <Grid
            gap={'size-100'}
            width={'100%'}
            maxWidth={'560px'}
            alignItems={'center'}
            justifyItems={'center'}
            columns={['30px', '1fr', '30px', '1fr', '30px']}
            areas={['step1 divider1 step2 divider2 step3', 'label1 . label2 . label3']}
        >
            <div
                className={classes.step}
                aria-label={'step one'}
                data-content={'1'}
                data-active={isUploadingOrPreparing(currentStep)}
                data-completed={isTaskTypeSelection(currentStep) || isLabelMapping(currentStep)}
                style={{ gridArea: 'step1' }}
            />
            <Divider width={'100%'} size={'M'} margin={'auto'} gridArea={'divider1'} />
            <div
                className={classes.step}
                aria-label={'step two'}
                data-content={'2'}
                data-active={isTaskTypeSelection(currentStep)}
                data-completed={isLabelMapping(currentStep)}
                style={{ gridArea: 'step2' }}
            />
            <Divider width={'100%'} size={'M'} margin={'auto'} gridArea={'divider2'} />
            <div
                className={classes.step}
                aria-label={'step three'}
                data-content={'3'}
                data-active={isLabelMapping(currentStep)}
                style={{ gridArea: 'step3' }}
            />

            <Text gridArea={'label1'}>{t('project.import.progress.dataset')}</Text>
            <Text gridArea={'label2'} UNSAFE_style={{ width: dimensionValue('size-900'), textAlign: 'center' }}>
                {t('project.import.progress.taskType')}
            </Text>
            <Text gridArea={'label3'}>{t('project.import.progress.labels')}</Text>
        </Grid>
    );
};

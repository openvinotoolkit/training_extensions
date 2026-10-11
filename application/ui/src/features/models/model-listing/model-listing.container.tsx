// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { NoTrainedModels } from '@/assets/illustrations';
import { useTranslation } from '@/i18n';
import { dimensionValue, Divider, Flex, Heading } from '@geti-ui/ui';
import { useGetCurrentRunningJobs } from 'hooks/api/jobs/jobs.hook';
import { usePrefetchModels } from 'hooks/api/use-get-models.hook';
import { isEmpty } from 'lodash-es';

import { usePrefetchTaskModelArchitectures } from '../hooks/api/use-get-model-architectures.hook';
import { TrainModel } from '../train-model/train-model.component';
import { Header } from './components/header.component';
import { CurrentRunningJobs } from './current-running-jobs/current-running-jobs.component';
import { ModelListing } from './model-listing.component';
import { ModelListingProvider, useModelListing } from './provider/model-listing-provider';

type ModelListingContainerProps = {
    exportJobs?: ReactNode;
};

const ModelListingContent = ({ exportJobs }: ModelListingContainerProps) => {
    const { t } = useTranslation();
    const runningJobs = useGetCurrentRunningJobs();
    const { groupedModels, searchBy, datasetRevisions, showFailedModels } = useModelListing();

    const hasNoResults = groupedModels.length === 0 && (searchBy.length > 0 || !showFailedModels);
    const hasNoModels = groupedModels.length === 0 && searchBy.length === 0 && showFailedModels;

    if (hasNoModels && isEmpty(runningJobs)) {
        return (
            <Flex
                direction={'column'}
                height={'100%'}
                alignItems={'center'}
                justifyContent={'center'}
                UNSAFE_style={{ padding: dimensionValue('size-300') }}
            >
                <CurrentRunningJobs datasetRevisions={datasetRevisions} />

                <Flex
                    direction={'column'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    gap={'size-100'}
                    marginTop={'size-600'}
                    flex={1}
                >
                    <NoTrainedModels />
                    <Heading level={2} UNSAFE_style={{ textAlign: 'center' }}>
                        {t('models.list.empty.title')}
                        <br />
                        {t('models.list.empty.description')}
                    </Heading>
                    <TrainModel />
                </Flex>
            </Flex>
        );
    }

    return (
        <Flex direction={'column'} height={'100%'} UNSAFE_style={{ padding: dimensionValue('size-300') }}>
            <Header />

            <Divider size={'S'} marginY={'size-300'} />

            <Flex direction={'column'} flex={1} UNSAFE_style={{ overflowY: 'auto', scrollbarGutter: 'stable' }}>
                <CurrentRunningJobs datasetRevisions={datasetRevisions} />

                {exportJobs}

                <ModelListing hasNoResults={hasNoResults} groupedModels={groupedModels} />
            </Flex>
        </Flex>
    );
};

export const ModelListingContainer = ({ exportJobs }: ModelListingContainerProps) => {
    // Start both requests in parallel; otherwise architectures only load after the provider's models query resolves
    usePrefetchModels();
    usePrefetchTaskModelArchitectures();

    return (
        <ModelListingProvider>
            <ModelListingContent exportJobs={exportJobs} />
        </ModelListingProvider>
    );
};

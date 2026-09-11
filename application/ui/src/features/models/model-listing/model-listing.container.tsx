// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { dimensionValue, Divider, Flex, Heading } from '@geti-ui/ui';
import { useGetCurrentRunningJobs } from 'hooks/api/jobs/jobs.hook';
import { isEmpty, isString } from 'lodash-es';

import { ReactComponent as NoTrainedModels } from '../../../assets/no-trained-models.svg';
import { ExportJobsList } from '../../dataset/import-export/export-jobs-list/export-jobs-list.component';
import { TrainModel } from '../train-model/train-model.component';
import { Header } from './components/header.component';
import { CurrentRunningJobs } from './current-running-jobs/current-running-jobs.component';
import { ModelListing } from './model-listing.component';
import { ModelListingProvider, useModelListing } from './provider/model-listing-provider';

const ModelListingContent = () => {
    const { t } = useTranslation();
    const runningJobs = useGetCurrentRunningJobs();
    const { groupedModels, searchBy, datasetRevisions, groupBy, showFailedModels } = useModelListing();

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
                <CurrentRunningJobs groupBy={groupBy} datasetRevisions={datasetRevisions} />

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
                <CurrentRunningJobs groupBy={groupBy} datasetRevisions={datasetRevisions} />

                <ExportJobsList predicate={({ datasetId }) => isString(datasetId)} />

                <ModelListing hasNoResults={hasNoResults} groupedModels={groupedModels} />
            </Flex>
        </Flex>
    );
};

export const ModelListingContainer = () => {
    return (
        <ModelListingProvider>
            <ModelListingContent />
        </ModelListingProvider>
    );
};

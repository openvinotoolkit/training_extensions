// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { useTranslation } from '@/i18n';
import { Button, Flex, Heading } from '@geti-ui/ui';
import { ENTIRE_DATASET_VIEW_ID, useDatasetViewId } from 'hooks/use-dataset-view-id.hook';

import { ReactComponent as EmptyDatasetImage } from '../../../assets/empty-dataset.svg';
import { useImportDatasetDialogState } from '../providers/export-import-dataset-dialog-provider.component';
import { ENTIRE_DATASET_NAME } from './toolbar/dataset-view-selector/util';
import { MediaUpload } from './toolbar/media-upload.component';

const ImportDatasetButton = () => {
    const { t } = useTranslation();
    const { datasetImportDialogState } = useImportDatasetDialogState();

    return (
        <Button variant={'secondary'} onPress={() => datasetImportDialogState.open()}>
            {t('dataset.empty.importDataset')}
        </Button>
    );
};

const EmptyMessage = ({ children }: { children: ReactNode }) => {
    return (
        <Heading level={2} UNSAFE_style={{ textAlign: 'center' }}>
            {children}
        </Heading>
    );
};

const NoMatchingMediaItems = () => {
    const { t } = useTranslation();
    return (
        <EmptyMessage>
            {t('dataset.empty.noMatches')}
            <br />
            {t('dataset.empty.changeFilter')}
        </EmptyMessage>
    );
};

const EmptyDatasetView = () => {
    const { t } = useTranslation();
    const [, setDatasetViewId] = useDatasetViewId();

    return (
        <>
            <EmptyMessage>
                {t('dataset.empty.viewEmpty')}
                <br />
                {t('dataset.empty.assignOrBack')}
            </EmptyMessage>
            <Button variant={'secondary'} onPress={() => setDatasetViewId(ENTIRE_DATASET_VIEW_ID)}>
                {t('dataset.empty.goToEntireDataset', { datasetName: ENTIRE_DATASET_NAME })}
            </Button>
        </>
    );
};

const EmptyEntireDataset = () => {
    const { t } = useTranslation();
    return (
        <>
            <EmptyMessage>
                {t('dataset.empty.datasetEmpty')}
                <br />
                {t('dataset.empty.uploadToStart')}
            </EmptyMessage>
            <Flex gap={'size-100'}>
                <MediaUpload testId={'upload-media-input-empty-dataset'} />
                <ImportDatasetButton />
            </Flex>
        </>
    );
};

type EmptyDatasetProps = {
    hasActiveFilter: boolean;
};

const EmptyDatasetContent = ({ hasActiveFilter }: EmptyDatasetProps) => {
    const [datasetViewId] = useDatasetViewId();

    if (hasActiveFilter) {
        return <NoMatchingMediaItems />;
    }

    if (datasetViewId !== ENTIRE_DATASET_VIEW_ID) {
        return <EmptyDatasetView />;
    }

    return <EmptyEntireDataset />;
};

export const EmptyDataset = ({ hasActiveFilter }: EmptyDatasetProps) => {
    return (
        <Flex direction={'column'} gap={'size-200'} alignItems={'center'} justifyContent={'center'} height={'100%'}>
            <EmptyDatasetImage />
            <EmptyDatasetContent hasActiveFilter={hasActiveFilter} />
        </Flex>
    );
};
